"""
Geometry container base class.




"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import os
from yt.extern.six.moves import cPickle
import weakref
from collections import OrderedDict
from yt.utilities.on_demand_imports import _h5py as h5py
import numpy as np

from yt.config import ytcfg
from yt.funcs import iterable
from yt.units.yt_array import \
    YTArray, uconcatenate
from yt.utilities.io_handler import io_registry
from yt.utilities.logger import ytLogger as mylog
from yt.utilities.parallel_tools.parallel_analysis_interface import \
    ParallelAnalysisInterface, parallel_root_only
from yt.utilities.exceptions import YTFieldNotFound

class IndexManager:
    def __init__(self, dataset):
        self.ds = self.dataset = weakref.proxy(ds)

        self.indices = OrderedDict()

    def add_index(self, index, name = None):
        if name is None:
            name = "index_%02i" % (len(self.indices))
        self.indices[name] = index

    def _split_fields(self, fields):
        # This will split fields into either generated or read fields
        fields_to_read, fields_to_generate = [], []
        for ftype, fname in fields:
            if fname in self.field_list or (ftype, fname) in self.field_list:
                fields_to_read.append((ftype, fname))
            elif fname in self.ds.derived_field_list or (ftype, fname) in self.ds.derived_field_list:
                fields_to_generate.append((ftype, fname))
            else:
                raise YTFieldNotFound((ftype,fname), self.ds)
        return fields_to_read, fields_to_generate

    def chunk_iter(self, dobj, chunk_type, chunk_args = None):
        chunk_args = chunk_args or {}
        # We iterate over each of the chunks.  The only reason we would need to
        # initialize a base chunk for them would be if we used that as a
        # basis for later subdividing, which I think we can get around.
        for name, index in self.indices.items():
            yield from index.chunk_iter(dobj, chunk_type, chunk_args)

    def chunk_sizes(self, dobj, chunk_type, chunk_args = None):
        for name, index in self.indices.items():
            yield from index.chunk_sizes(dobj, chunk_type, chunk_args)

class Index(ParallelAnalysisInterface):
    """The base index class"""
    _unsupported_objects = ()
    _index_properties = ()

    def __init__(self, ds, dataset_type):
        ParallelAnalysisInterface.__init__(self)
        self.dataset = weakref.proxy(ds)
        self.ds = self.dataset

        self._initialize_state_variables()

        mylog.debug("Setting up domain geometry.")
        self._setup_geometry()

        mylog.debug("Initializing data grid data IO")
        self._setup_data_io()

        # Note that this falls under the "geometry" object since it's
        # potentially quite expensive, and should be done with the indexing.
        mylog.debug("Detecting fields.")
        self._detect_output_fields()

    def _initialize_state_variables(self):
        self._parallel_locking = False
        self._data_file = None
        self._data_mode = None
        self.num_grids = None

    def _setup_data_io(self):
        if getattr(self, "io", None) is not None: return
        self.io = io_registry[self.dataset_type](self.dataset)

    def _get_particle_type_counts(self):
        # this is implemented by subclasses
        raise NotImplementedError

    def _read_particle_fields(self, fields, dobj, chunk = None):
        if len(fields) == 0: return {}, []
        fields_to_read, fields_to_generate = self._split_fields(fields)
        if len(fields_to_read) == 0:
            return {}, fields_to_generate
        selector = dobj.selector
        if chunk is None:
            self._identify_base_chunk(dobj)
        chunks = self._chunk_io(dobj, cache = False)
        fields_to_return = self.io._read_particle_selection(
            chunks,
            selector,
            fields_to_read)
        return fields_to_return, fields_to_generate

    def _read_fluid_fields(self, fields, dobj, chunk = None):
        if len(fields) == 0: return {}, []
        fields_to_read, fields_to_generate = self._split_fields(fields)
        if len(fields_to_read) == 0:
            return {}, fields_to_generate
        selector = dobj.selector
        if chunk is None:
            self._identify_base_chunk(dobj)
            chunk_size = dobj.size
        else:
            chunk_size = chunk.data_size
        fields_to_return = self.io._read_fluid_selection(
            self._chunk_io(dobj),
            selector,
            fields_to_read,
            chunk_size)
        return fields_to_return, fields_to_generate

    def chunk_iter(self, dobj, chunk_type, ngz = 0, **kwargs):
        # A chunk is either None or (grids, size)
        if dobj._current_chunk is None:
            self._identify_base_chunk(dobj)
        if ngz != 0 and chunk_type != "spatial":
            raise NotImplementedError
        if chunk_type == "all":
            return self._chunk_all(dobj, **kwargs)
        elif chunk_type == "spatial":
            return self._chunk_spatial(dobj, ngz, **kwargs)
        elif chunk_type == "io":
            return self._chunk_io(dobj, **kwargs)
        else:
            raise NotImplementedError

def cached_property(func):
    n = '_%s' % func.__name__
    def cached_func(self):
        if self._cache and getattr(self, n, None) is not None:
            return getattr(self, n)
        if self.data_size is None:
            tr = self._accumulate_values(n[1:])
        else:
            tr = func(self)
        if self._cache:

            setattr(self, n, tr)
        return tr
    return property(cached_func)

class YTDataChunk(object):

    def __init__(self, dobj, chunk_type, objs, data_size = None,
                 field_type = None, cache = False, fast_index = None):
        self.dobj = dobj
        self.chunk_type = chunk_type
        self.objs = objs
        self.data_size = data_size
        self._field_type = field_type
        self._cache = cache
        self._fast_index = fast_index

    def _accumulate_values(self, method):
        # We call this generically.  It's somewhat slower, since we're doing
        # costly getattr functions, but this allows us to generalize.
        mname = "select_%s" % method
        arrs = []
        for obj in self._fast_index or self.objs:
            f = getattr(obj, mname)
            arrs.append(f(self.dobj))
        if method == "dtcoords":
            arrs = [arr[0] for arr in arrs]
        elif method == "tcoords":
            arrs = [arr[1] for arr in arrs]
        arrs = uconcatenate(arrs)
        self.data_size = arrs.shape[0]
        return arrs

    @cached_property
    def fcoords(self):
        if self._fast_index is not None:
            ci = self._fast_index.select_fcoords(
                self.dobj.selector, self.data_size)
            ci = YTArray(ci, input_units = "code_length",
                         registry = self.dobj.ds.unit_registry)
            return ci
        ci = np.empty((self.data_size, 3), dtype='float64')
        ci = YTArray(ci, input_units = "code_length",
                     registry = self.dobj.ds.unit_registry)
        if self.data_size == 0: return ci
        ind = 0
        for obj in self._fast_index or self.objs:
            c = obj.select_fcoords(self.dobj)
            if c.shape[0] == 0: continue
            ci[ind:ind+c.shape[0], :] = c
            ind += c.shape[0]
        return ci

    @cached_property
    def icoords(self):
        if self._fast_index is not None:
            ci = self._fast_index.select_icoords(
                self.dobj.selector, self.data_size)
            return ci
        ci = np.empty((self.data_size, 3), dtype='int64')
        if self.data_size == 0: return ci
        ind = 0
        for obj in self._fast_index or self.objs:
            c = obj.select_icoords(self.dobj)
            if c.shape[0] == 0: continue
            ci[ind:ind+c.shape[0], :] = c
            ind += c.shape[0]
        return ci

    @cached_property
    def fwidth(self):
        if self._fast_index is not None:
            ci = self._fast_index.select_fwidth(
                self.dobj.selector, self.data_size)
            ci = YTArray(ci, input_units = "code_length",
                         registry = self.dobj.ds.unit_registry)
            return ci
        ci = np.empty((self.data_size, 3), dtype='float64')
        ci = YTArray(ci, input_units = "code_length",
                     registry = self.dobj.ds.unit_registry)
        if self.data_size == 0: return ci
        ind = 0
        for obj in self._fast_index or self.objs:
            c = obj.select_fwidth(self.dobj)
            if c.shape[0] == 0: continue
            ci[ind:ind+c.shape[0], :] = c
            ind += c.shape[0]
        return ci

    @cached_property
    def ires(self):
        if self._fast_index is not None:
            ci = self._fast_index.select_ires(
                self.dobj.selector, self.data_size)
            return ci
        ci = np.empty(self.data_size, dtype='int64')
        if self.data_size == 0: return ci
        ind = 0
        for obj in self._fast_index or self.objs:
            c = obj.select_ires(self.dobj)
            if c.shape == 0: continue
            ci[ind:ind+c.size] = c
            ind += c.size
        return ci

    @cached_property
    def tcoords(self):
        self.dtcoords
        return self._tcoords

    @cached_property
    def dtcoords(self):
        ct = np.empty(self.data_size, dtype='float64')
        cdt = np.empty(self.data_size, dtype='float64')
        self._tcoords = ct # Se this for tcoords
        if self.data_size == 0: return cdt
        ind = 0
        for obj in self._fast_index or self.objs:
            gdt, gt = obj.select_tcoords(self.dobj)
            if gt.size == 0: continue
            ct[ind:ind+gt.size] = gt
            cdt[ind:ind+gdt.size] = gdt
            ind += gt.size
        return cdt

    @cached_property
    def fcoords_vertex(self):
        nodes_per_elem = self.dobj.index.meshes[0].connectivity_indices.shape[1]
        dim = self.dobj.ds.dimensionality
        ci = np.empty((self.data_size, nodes_per_elem, dim), dtype='float64')
        ci = YTArray(ci, input_units = "code_length",
                     registry = self.dobj.ds.unit_registry)
        if self.data_size == 0: return ci
        ind = 0
        for obj in self.objs:
            c = obj.select_fcoords_vertex(self.dobj)
            if c.shape[0] == 0: continue
            ci[ind:ind+c.shape[0], :, :] = c
            ind += c.shape[0]
        return ci


class ChunkDataCache(object):
    def __init__(self, base_iter, preload_fields, geometry_handler,
                 max_length = 256):
        # At some point, max_length should instead become a heuristic function,
        # potentially looking at estimated memory usage.  Note that this never
        # initializes the iterator; it assumes the iterator is already created,
        # and it calls next() on it.
        self.base_iter = base_iter.__iter__()
        self.queue = []
        self.max_length = max_length
        self.preload_fields = preload_fields
        self.geometry_handler = geometry_handler
        self.cache = {}

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if len(self.queue) == 0:
            for i in range(self.max_length):
                try:
                    self.queue.append(next(self.base_iter))
                except StopIteration:
                    break
            # If it's still zero ...
            if len(self.queue) == 0: raise StopIteration
            chunk = YTDataChunk(None, "cache", self.queue, cache=False)
            self.cache = self.geometry_handler.io._read_chunk_data(
                chunk, self.preload_fields) or {}
        g = self.queue.pop(0)
        g._initialize_cache(self.cache.pop(g.id, {}))
        return g
