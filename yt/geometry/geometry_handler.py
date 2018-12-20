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
from yt.utilities.data_chunk import YTDataChunk

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

    def chunk_sizes_exact(self, dobj, chunk_type, chunk_args = None):
        for name, index in self.indices.items():
            yield from index.chunk_sizes_exact(dobj, chunk_type, chunk_args)

    def chunk_sizes_upper(self, dobj, chunk_type, chunk_args = None):
        for name, index in self.indices.items():
            yield from index.chunk_sizes_upper(dobj, chunk_type, chunk_args)

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

