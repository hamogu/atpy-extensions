from operator import itemgetter
import numpy as np
import copy
import warnings

import sys
sys.path.append("/data/hguenther/soft/lib/python")
import os

import coords

import asciitable
import atpy


# TBD add support for non-scalars in add_source etc.
# TBD check for CamelCase
class Table(atpy.Table):
    '''Extends AtPy.Table functionalty to single rows
    
    ATPy is build around handling columns - it can add, rename, delete columns etc.
    While direct data acess is to each cell in the table is possible,
    AtPy provides no special functions to do so.
    This extension to ATPy adds mechanisms to change values in rows and adds
    some convenience functions to make Table look like a ndarray from the outside 
    in some respects.
    
    In the current version it does not support all the generality of ATPy, it misses:
      * columns where each element is a vector
      * masked arrays'''
    
    character_null = ''
    float_null = np.nan
    integer_null = 0
    complex_null = np.complex(np.nan)
    
    def __init__(self, *args, **kwargs):
        atpy.Table.__init__(self, *args, **kwargs)
        
        # if some data is loaded during the initialisation
        if self.data != None:
            # check the null value - some formats don't contain one and then atpy might not set it the way we want
            for name in self.names:
                try:
                    test=np.array( self.columns[name].null, dtype=self[name].dtype)
                except ValueError:
                    self.columns[name].__dict__['null']=self.default_null(self[name].dtype)
                    warnings.warn("WARNING: null value of column %s has been reset to default." % name)
    
    def keys(self):
        '''An alias for Table.data.dtype.names
        
        keys() is a standard function to look for keys to iterate over, so we supply that.'''
        try: 
		names = self.data.dtype.names
		return names
	except AttributeError:
		return []

    def default_null(self, dtype):
        '''Return the default null for the datatype dtype.
        
        The class defines the values character_null, float_null, integer_null, complex_null
	Given a datatyoe this function looks up and returns the default null type.
	The default types can be changed by changing the class attributes
	character_null, float_null, integer_null and complex_null
	'''
	#TBD: Can this maybe implemented easier with a dictionary?
        if np.issubdtype(dtype, np.character):  # np.character contains str and unicode
            return self.character_null
        elif np.issubdtype(dtype, float):
            return self.float_null
        elif np.issubdtype(dtype, np.integer):  # np.integer contains intXX and uintXX
            return self.integer_null
        elif np.issubdtype(dtype, complex):
            return self.complex_null
        else:
            raise ValueError('No default null value for dtype %s' % dtype )
    
    def add_empty_column(self, name, dtype, **kwargs):
	if not 'null' in kwargs: kwargs['null'] = self.default_null(dtype)
	atpy.Table.add_empty_column(self, name, dtype, **kwargs)
	self.data[name][:] = self.columns[name].null
    
    def _fields(self, data):
        '''This internal method finds the namelist of common subscriptable objects
        
        Several possible objects can be used to input data into a catalog, however they all 
        need to supply a list of names which can be matched to the column names of the catalog.
        This method provides a unified interface to get the names from
        * ATPy tables (or a catalog that inherits from ATPy.table)
        * numpy record arrys
        * dictionaries
        * an object that implements the keys() method
        '''
        try: 
            keys = data.dtype.names
            return keys
        except AttributeError:
            try: 
                keys = data.keys()
                return keys
            except AttributeError:
                raise TypeError('No rule found to match fields to input keys')
    
    def auto_add_columns(self, data, auto_add_columns):
	'''This method adds columns to the table based on the data found in data
	
	This method detects the keyword in data and then adds columns to the table.
	Input:
	    data: can be any object, where _fields can detect the keywords.
	    auto_add_columns: If true columns for all keys in data, which are 
	        not yet in the table, are added
		If it is a string or a list of strings, only those columns are
		added, provided they are not contained in the table.
	'''
        if auto_add_columns == True:
            auto_add_columns = self._fields(data)
        if isinstance(auto_add_columns, basestring):
            auto_add_columns = [ auto_add_columns ]
        set_auto_add_columns = set(auto_add_columns)
        set_data_dtype_names = set(self._fields(data))
        if not set_auto_add_columns.issubset(set_data_dtype_names):
            raise ValueError(str(set_auto_add_columns - set_data_dtype_names) + ' is not contained in data!')
        # If self.data in not empty, take out those names which already exist
	if (isinstance(self.data, type(None))) or len(self.data) == 0:
		nlines = 1 #If table is still empty, add 1 line
	else:
		#if not len(self.data) > 0: set_auto_add_columns = set_auto_add_columns - set(self.names)
		#if self.data: shape = len(self.data)
		set_auto_add_columns = set_auto_add_columns - set(self.names)
		nlines = len(self.data)
        for columnname in list(set_auto_add_columns):
            # The column we add could be a vector column of vectors, so copy its shape
            # Not all objects have a shape attribute, so convert to np.array first
            shape = list(np.array(data[columnname]).shape)  # tuples are immutable -> convert to list
            if shape:
                shape[0] = nlines   # first element of shape is the length of the table.
                shape=tuple(shape)  # make sure shape is tuple. Maybe this could be left to auto typecasting, but I want it to look as it should
            else:                   # for arrays with only one element shape might be ()
                shape = (1,)
            if isinstance(data, atpy.Table):
                self.add_empty_column(columnname,data[columnname].dtype, unit = data.columns[columnname].unit, null=data.columns[columnname].null, description=data.columns[columnname].description, format=data.columns[columnname].format, shape = shape)
            elif isinstance(data, np.recarray):
                self.add_empty_column(columnname,data[columnname].dtype, shape = shape)
            else: 
                dtype=np.array(data[columnname]).dtype
                null=self.default_null(dtype)
                self.add_empty_column(columnname, dtype, null=null, shape = shape)
    
    def empty_row(self):
	'''This method return an empty row for the table, based on columns.null
	'''
        if len(self) == 0:
            newrow = None
        else:
            newrow=self.rows([0])
            for name in newrow.names:
                newrow[name][0]=newrow.columns[name].null
        return newrow
    
    def add_row(self, data, auto_add_columns = False):
	'''This method adds a row to the table.
	
	A similar functionalty can be achived with append, but add_row is more 
	general. It first creates an empty_row and fills only those values
	for which keys are found in data.
	Input:
	    data: can be any object, where _fields can detect the keywords.
	    auto_add_columns: If true columns for all keys in data, which are 
	        not yet in the table, are added
		If it is a string or a list of strings, only those columns are
		added, provided they are not contained in the table.
	'''
        if self.data == None: add_first_row_in_table = True  #if self.data does not work -> "truth value of an array in ambiguous"
	else: add_first_row_in_table = False
	if auto_add_columns != False:
            self.auto_add_columns(data,auto_add_columns)
        newrow=self.empty_row()
        for datum in list(set(self._fields(data)) & set(self.keys())):
            newrow.data[datum] = data[datum]
	#if this is the first row in a new table, then auto_add_columns will add that row
	# so an update of row zero is all we want
        if add_first_row_in_table: self.update_row(0, newrow, overwrite = True)
	else: self.append(newrow)
    
    def update_row(self, rowindex, data, auto_add_columns = False, overwrite = False):
	'''Modifies the values of a specific row in the table under certain conditions.
	
	Input:
	    rowindex: index of row to be changed
	    data: can be any object, where _fields can detect the keywords.
	    auto_add_columns: If true columns for all keys in data, which are 
	        not yet in the table, are added
		If it is a string or a list of strings, only those columns are
		added, provided they are not contained in the table.
	    overwrite = False: The method only changes those columns which are
	        new (from auto_add_columns) or equal the null value in the table.
		If true the values for all keys in data are replaced. 
		If overwrite is a string or a list of strings, only those
		columns (and those with null values) are overwritten in the table.
		For vector columns overwrite either leaves the full vector in a row
		unchanged or overwrites the full vector (not just single elements).
	'''	
        if auto_add_columns != False:
            self.auto_add_columns(data,auto_add_columns)
        if overwrite:
            if overwrite == True: overwritenames = set(self._fields(data)) #overwrite = True
            else: overwritenames = set(overwrite)  #overwrite is a list of column names
        else: overwritenames = set([ ])
        # if an element equals self.columns[name].null it can be overwritten
        # a special case is needed for nan - they cannot be found by equality testing
        # also np.isnan() only works for numbers -> hide this in try catch block
        # 
        # enclose all tests in np.all, so that it equally works for vector columns
        for name in self.keys(): 
            if np.all(self.columns[name].null == self.data[name][rowindex]): overwritenames = overwritenames | set([name])
            try:
                if (np.isnan(self.columns[name].null) and np.all(np.isnan(self.data[name][rowindex]))): overwritenames = overwritenames | set([name])
            except NotImplementedError:
                pass  # if np.nan fails, this is a string, so do nothing
        #restrict overwritenames to those columns which are actually in the input data
        overwritenames = list( overwritenames & set(self._fields(data)) )
        for name in overwritenames:
            if isinstance(data[name], np.ndarray): self.data[name][rowindex] = data[name][0]
	    else: self.data[name][rowindex] = data[name]
    
    def __getitem__(self, item):
	    '''make the ATPy table directly indexable (table['a']) as a shorthand for table.data['a']
	    '''
            return self.data[item]


class SimpleCoords(object):
    '''A simple class to handle a column of coordinates for a catalog.
    
    This class is not very sophisticated, in fact it is the simplest possible
    implementation of a Coords class for BaseCatalog.
    A lot of overhead is spend to make it look like a ndarray from the outside.
    However, abstraction into an extra class makes
        1) a well-defined interface to the table in the catalog
	2) More sophisticated CoordTables can be implented,
	   as long as they provide the same interface,
	   e.g. organizing the coordinates in a KD-tree to faster searching
	   or interfacing a C module to do that.

    SimpleCoords is good for Catalogs with < 1000 entries, for situations, 
    when search time is not an issue. The search is implemented in Python,
    thus its internals are redily avaiable, which is great for debugging or
    customizing.
    
    There are two levels to acess the coordinates in a table:
        1) On the highly abstracted level routines return
	   the index of the nearest neigbour or all sources within
	   a given distance.
	2) On the lower level the coordinates themselves can be read out as
	   SimpleCoords[2:4]. The return values are always of type 
	   coords.Position (module coords, s. below)
	   To set coordinates, they need to be transformed to that type.
	   The function calc is provided to do just that.
	3) On the lowest level Python allows access to all internals at own risk :-)
    
    SimpleCoords uses the coords package to represent coordinates and
    angular distances. 
     
    SimpleCoords uses coords which is Copyright (C) 2006 
    Association of Universities for Research in Astronomy (AURA)
    See coords package for full license. 
    
    Coords package is available from:
    https://www.stsci.edu/trac/ssb/astrolib/wiki
    '''
    def __init__(self, pos = None):
        if pos: self.table = pos
        else: self.table=np.array([],dtype=object)
        self.units='degrees'
	self.angsepunits='arcsec'
    
    
    ### Convenience functions to make coordinates as easy to use as a Python sequence ###
    def __getitem__(self, item):
	    if np.array(item).size == 0:
                return []
            #elif np.array(self.table).size == 1:
            #    return self.table
            else: return self.table[item]
    
    def __setitem__(self, key, value):
        if (key == 0) and (len(self.table) == 0): self.table= np.array([value],dtype=object)
	else: self.table[key]=value
    
    def __len__(self):
        return len(self.table)
    
    def set(self,pos):
	'''Replaces the full table of coordinates.
	
	Input:
	    pos: list or np.ndarray of coord.Position values
	'''
        if isinstance(pos, np.ndarray): self.table = pos
        else: self.table = np.array(pos)
    
    def __str__(self):
        str(self.table)
    
    
    ### Functions to change set of coordinates ###
    def calc(self, pos, units=None):
	'''Transform values of different forms to coord.Position type
	
	pos: coordinates
	See coord package for full specification of input types.
	Accepted are e.g.
	    * tuples of RA, DEC: (22.8,56.8)
	    * strings: "11:34:12.5 +56:56:23.5"
	    * units can be "degrees" or "radian" or "hmsdms"
	
	The method is provided for medium level access to convert any input to
	internal coordinates.
	
	unit [default: self.units]: unit of coordinates.
	    will be passed to coords.Position
	'''
        if not units: units=self.units
        return coords.Position(pos,units=units)
    
    def reset(self):
	'''Reset the table to empty, but keep all other attributes.'''
        self.table=np.array([],dtype=object)
    
    def append(self, pos):
	'''Append a list or np.ndarray of coordinates
	
	Coordinates need to be of type. coords.Position
	'''
        self.table=np.hstack((self.table, pos))
    
    
    ### Functions to search in the coordinates ###
    def distto(self,pos):
	'''Calculate an array of distances of all elements to position pos
	
	This ia medium-level acess routine. 
	It return a list of coords.AngSep objects.
	input:
	    pos: coords.Position objects or format, which can be converted
	'''
        if isinstance(pos, coords.Position): base = pos
        else: base=coords.Position(pos)
        return map(base.angsep, self.table)
    
    def NNindexdist(self, pos):
	'''Return index and distance of the entry closest to pos.
	
	This ia medium-level acess routine. 
	It returns an index and an coords.AngSep object.
	input:
	    pos: coords.Position objects or format, which can be converted
	'''
        dist=self.distto(pos)
        return min(enumerate(dist), key=itemgetter(1))
    
    def NNindex(self, pos, maxdist=-1., maxdistunits = None):
	'''Return index of the entry closest to pos.
	
	This is high-level acess routine. 
	It returns an index.
	input:
	    pos: coords.Position objects or format, which can be converted
	keywords:
	    maxdist: If no match within maxdist is found, None is returned.
	    	For negative values of maxdist the closest match is retunred,
		whatever its distance. [default = -1]
	    maxdist_units: ['degrees','arcsec' or 'radian'] 
	        unit of maxdist [default: self.angsepunits]
	    
	Required for use with a BaseCatalog object.
	'''
	if len(self) == 0: return None
	if not maxdistunits: maxdistunits = self.angsepunits
        NNindex,NNdist = self.NNindexdist(pos)
        if maxdist < 0. or coords.AngSep(maxdist, units = maxdistunits) >= NNdist:
            return NNindex
        else:
            return None
    
    def allwithin(self, pos, maxdist, maxdistunits = None):
	'''Return a list of indices of all entries within maxdist of pos.
	
	This is high-level acess routine. 
	It returns a list of indices. If no object is found, the list is empty.
	input:
	    pos: coords.Position objects or format, which can be converted
	    maxdist: If no match within maxdist is found, None is returned.
	    	For negative values of maxdist the closest match is retunred,
		whatever its distance. [default = -1]
	keywords:
	    maxdist_units: ['degrees','arcsec' or 'radian'] 
	        unit of maxdist [default: self.angsepunits]
	    
	Required for use with a BaseCatalog object.
	'''

        dist=self.distto(pos)
        maxd=coords.AngSep(maxdist)
        index= np.array(map(lambda d: maxd >= d, dist)).nonzero()[0]
        return index


#Different typical functions to make coordinates
def coord_RADEC(self, data):
    return zip(data[self.ra],data[self.dec])

def coord_RADECstring(self, data):
    return map(lambda line: str(line[0])+' '+str(line[1]),zip(data[self.ra], data[self.dec]))

class BaseCatalog(Table):
    
    def __init__(self, *args, **kwargs):
        self.ra = 'RA'
        self.dec = 'DEC'
        self.form_coords = coord_RADEC
        if 'Coords' in kwargs: self.coords = (kwargs.pop('Coords'))
        else: self.coords = SimpleCoords()
        Table.__init__(self, *args, **kwargs)
	self.match_source = match_source
    
    ### Functions to make self.coords ###
    def form_coords(self, data):
        raise NotImplementedError('You need to override form_coords with a function to form coordinates from the data!')
    
    def calc_all_coords(self):
        self.coords.set(map(self.coords.calc,self.form_coords(self, self.data)))
    ### Extension of ATPy functions which effect self.coords
    
    def read(self, *args, **kwargs):
        if 'RA' in kwargs:
            self.ra = kwargs.pop('RA')
        if 'DEC' in kwargs:
            self.dec = kwargs.pop('DEC')
        if 'form_coords' in kwargs:
            self.form_coords = kwargs.pop('form_coords')
        Table.read(self, *args, **kwargs)
	self.calc_all_coords()
    
    def add_column(self, name, data,  **kwargs):
        '''calc coords if second column of RA DEC has been added
        
        This prescription only covers the standard case of columns with self.RA and self.DEC
        If self.form_coords is overridden by a more complex prescrition calc_all_coords()
        may have to be called explicitly by the user.
        '''
        Table.add_column(self, name, data,  **kwargs)

        if name in (self.ra, self.dec) and self.ra in self.names and self.dec in self.names and \
            (self.data[name] <> np.zeros(self.__len__(), dtype=self.data[name].dtype)).all():
            self.calc_all_coords()
    
    def sort(self, keys):
        '''Sort the table according to one or more keys.
        
	This operates on the existing table (and does not return a new table).
        Inputs:
            keys: [ string | list of strings ]
                The key(s) used for reordering
        '''
        if not type(keys) == list:
            keys = [keys]
        index = np.argsort(self.data, order=keys)
        self.data = self.data[index]
	self.coords[:] = self.coords[index] 
    
    def reset(self):
        '''Empty the table, but keep functions, e.g. calc_coord'''
        Table.reset(self)
        self.coords.reset()
    
    def append(self, newdata, pos=None):
        Table.append(self, newdata)
        if pos:
            self.coords.append(pos)
        else:
            self.coords.append(map(self.coords.calc,self.form_coords(self, newdata)))
    
    
    ### direct extension of ATPy functions which require adding self.coords
    def where(self, mask):
        new_table = Table.where(self, mask)
        new_table.coords = copy.copy(self.coords)
        if self.coords: new_table.coords.set(self.coords[mask]) #set, if list is not empty
        new_table.ra = self.ra
        new_table.dec = self.dec
        new_table.form_coords = self.form_coords
        new_table.match_source = self.match_source
        return new_table
    
    def row(self, row_number, python_types = False):
        row = Table.row(self, row_number, python_types = python_types)
        return row, self.coords[row_number]
    
    
    ### indirect extenstion of ATPy - functions defined in Table
    def empty_row(self):
        emptyrow = Table.empty_row(self)
        if emptyrow: emptyrow.coords.reset()
        return emptyrow
    
    def update_row(self, rowindex, data, auto_add_columns = False, overwrite = False):
        Table.update_row(self, rowindex, data, auto_add_columns = auto_add_columns, overwrite = overwrite)
        #update coordinates is necessary
        if overwrite:
            if overwrite == True: overwritenames = set(self._fields(data)) #overwrite = True
            else: overwritenames = set(overwrite)  #overwrite is a list of columnnames
        else: overwritenames = set([ ])
        if (self.ra in overwritenames) or (self.dec in overwritenames):
            self.coords[rowindex] = self.coords.calc(self.form_coords(self, self.rows([rowindex]))[0])
    
    def add_row(self, data, pos = None, auto_add_columns = False):  #Do I really need to copy everything, just to get the pos=pos in there? 
        ''' If pos is given, it needs to be of coord type, if not use coords.calc first
	'''
	# The following line works because of short-circuting - NoneType has no len()!
	if  isinstance(self.data, type(None)) or len(self.data) == 0:
		add_first_row_in_table = True
		if not (self.ra in self._fields(data) and self.dec in self._fields(data)): raise Exception('First row added to the table must contain RA and DEC')
	else:
		add_first_row_in_table = False
	if auto_add_columns != False:
            self.auto_add_columns(data,auto_add_columns)
        newrow=self.empty_row()
        for datum in list(set(self._fields(data)) & set(self.keys())):
            newrow.data[datum] = data[datum]
	#if this is the first row in a new table, then auto_add_columns will add that row
	# so an update of row zero is all we want
        if add_first_row_in_table:
		if pos: self.coords[0] = pos #make empty coords[0]. Will be recalculated by update_row
		else: self.coords[0] = self.coords.calc(self.form_coords(self, data))
		self.update_row(0, newrow, overwrite = True)
		if pos: self.coords[0] = pos
	else: self.append(newrow, pos = pos)
    
    
    
    ### Functions to search in the catalog and retrieve certain data ###
    def NNdata(self, pos, maxdist=-1.):
        NNindex=self.coords.NNindex(pos, maxdist)
        if NNindex == None or self.data == None:
            return None
        else:
            return self.row(NNindex)
    
    def allwithin(self, pos, maxdist):
        index  = self.coords.allwithin(pos, maxdist)
        return self.rows(index)
    
    
    #Functions to identify matches in the catalog and add or merge sources
    def add_source_info(self,pos,data,auto_add_columns=False,overwrite = False):
        '''
        
        '''
        match_source=self.match_source(self, pos,data)
        if match_source == None:
            self.add_row(data, pos = pos, auto_add_columns = auto_add_columns)
        else:
            self.update_row(match_source, data, auto_add_columns = auto_add_columns, overwrite = overwrite)
    
    def add_catalog(self, pos, data, auto_add_columns = False, overwrite = False):
        for i in range(len(data)):
            self.add_source_info(pos[i], data[[i]], auto_add_columns = auto_add_columns, overwrite = overwrite)
    
    def update_rows(self, pos, data, auto_add_columns = False, overwrite = False, subset = None):
        '''Update rows in a catalog with information from a different catalog
        
        Contrary to add_catalog, update_rows does not add sources to the catalog, but only adds
        information to existing sources, e.g. add te JHK magnitudes from 2MASS to an X-ray catalog,
        but do not add those 2MASS sources without X-ray counterpart.
        
        The procedures searched for matching sources with self.match_source and fills empty fields
        in those sources which are identified as matches with information from data.
        
        This methods allows to pass an index array to operate only on a subset of the original catalog.
        
        input:
            pos: an array of coordintes
            data: a catalog woth properties that should be added.
                This can take the form of a BaseCatalog instance, a numpy.ndarray or a list of dictionaries
        	
        keywords:
            auto_add_columns: Specifies which, if any, fields shold be added to the original catalog.
                For a full listing of allowed syntaxes, see Table.auto_add_columns
            overwrite: Specifies which data fields should be overwritten with information from the new catalog,
                if they are already filled. SDame syntax as auto_add_columns.

        TBD: Should this go the other way round> Now: steps through data and looks for best match, but should it not really step through self?
        '''
        if subset.dtype == 'bool':
            subsetind, = subset.nonzero()
        elif index == None:
            subsetind = range(len(self.data))
        else:
            subsetind = index
        subsettable = self.where(subsetind)
        for i in range(len(data)):
            match_source=subsettable.match_source(subsettable,pos[i],data[i])
            if match_source != None:
                self.update_row(subsetind[match_source], data[i], auto_add_columns = auto_add_columns, overwrite = overwrite)
    
def match_source(self, pos, data):
    return self.coords.NNindex(pos)


class SimpleCatalog(BaseCatalog):
    '''A BaseCatalog initialized with SimpleCoords as coordinate table 
    '''
    def __init__(self, *args, **kwargs):
	kwargs['Coords'] = SimpleCoords()
        BaseCatalog.__init__(self, *args, **kwargs)
    def read(self, *args, **kwargs):
        if 'form_coords' in kwargs:
            self.form_coords = kwargs.pop('form_coords')
            if 'unit' in kwargs:
                self.coords.units = kwargs.pop('unit')
                if not self.coords.units.lower()[0:3] in ['deg', 'rad', 'hms']:
                    raise Exception('Choose unit only from [degree, radians, hmsdms]')
            else: raise Exception('User form_coords function needs to specify coordinate unit from [degree, radians, hmsdms]')
        else:
            if 'unit' in kwargs:
                coord_units = kwargs.pop('unit')
                if lower(coord_units).startswith('rad'):
                    self.form_coords = self.coordRADECrad
                    self.coords.units='radians'
                elif lower(coord_units).startswith('hms'):
                    self.form_coords = self.coordRADEChmsdms
                    self.coords.units='hmsdms'
        BaseCatalog.read(self, *args, **kwargs)
	
class IRAFStandardCatalog(BaseCatalog):
	def __init__(self, *args, **kwargs):
		if len(args) == 0: BaseCatalog.__init__(self)
		else: BaseCatalog.__init__(self, args[0], type = 'ascii', Coords = SimpleCoords(), form_coords = coord_RADECstring, Reader=asciitable.NoHeaderReader, **kwargs)
		# reading an ASCII table never gets the keys right
		for key in self.keys(): self.columns[key].__dict__['null'] = self.default_null(self.columns[key].dtype)

def match_source_name(self, pos, data):
	return np.where(self['name'] == pos)[0]

class StetsonStandardCatalog(BaseCatalog):


	def __init__(self, *args, **kwargs):
		if len(args) == 0: BaseCatalog.__init__(self)
		else:
			# no arg required in code, but need to put any dummy
			# because a call with no argument would create an empty dummy
			posfile = os.path.join(os.path.dirname(__file__),'Land.pos')
			magfile = os.path.join(os.path.dirname(__file__),'hans.dat')
			kwargs['type'] = 'ascii'
			kwargs['Reader'] = asciitable.NoHeader
			kwargs['names'] = ['Stand_RA', 'Stand_DEC', 'RAh', 'RAmin', 'RAsec', 'DECd', 'DECmin', 'DECsec', 'xoffset', 'yoffset', 'x', 'y', 'name']
			kwargs['include_names'] = ['Stand_RA', 'Stand_DEC', 'name']
			kwargs['RA'] = 'Stand_RA'
			kwargs['DEC'] = 'Stand_DEC'
			kwargs['form_coords'] = coord_RADEC
			BaseCatalog.__init__(self, posfile, **kwargs)
			names = ['name', 'var', 'V', 'V_err', 'B', 'B_err', 'I', 'I_err', 'R', 'R_err', 'U', 'U_err', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'string']
			include_names = ['name', 'V', 'V_err', 'B', 'B_err', 'I', 'I_err', 'R', 'R_err', 'U', 'U_err']
			mags = Table(magfile, type = 'ascii', data_start = 1, Reader = asciitable.NoHeader, names = names, include_names = include_names)
			#now merge the magnitude data from hans.dat with the catalog with the coordinate info
			self.match_source = match_source_name
			self.add_catalog(mags['name'], mags.data, auto_add_columns=True)
			# reading an ASCII table never gets the keys right
			for key in ['V', 'B', 'I', 'R', 'U']: self[key][self[key] > 99.99] = self.columns[key].null
			for key in ['V_err', 'B_err', 'I_err', 'R_err', 'U_err']: self[key][self[key] > 9.99] = self.columns[key].null

class CartesianCoords(SimpleCoords):
    '''A simple class to handle a column of coordinates for a catalog.
    
    A modification of SimpleCoords. This works with cartesian Coords, e.g. 
    pixels in an image. Even simpler than SimpleCoords, so really this should be the BaseClass,
    but I figure that is will be used less in practice.
    '''
    def __init__(self, pos = None):
        if pos: self.table = pos
        else: self.table=np.array([],dtype=object)
    
    
    ### Functions to change set of coordinates ###
    def calc(self, pos):
	'''Do nothing. This is just an placeholder for more complicated Coords.
	'''
        return pos
    
    
    ### Functions to search in the coordinates ###
    def distto(self,pos):
	'''Calculate an array of distances of all elements to position pos
	
	This ia medium-level acess routine. 
	'''
        return map(lambda xy: ((pos[0]-xy[0])**2+(pos[1]-xy[1])**2)**0.5, self.table)
    
    def NNindex(self, pos, maxdist=-1.):
	'''Return index of the entry closest to pos.
	
	This is high-level acess routine. 
	It returns an index.
	input:
	    pos: Position as (x,y)
	keywords:
	    maxdist: If no match within maxdist is found, None is returned.
	    	For negative values of maxdist the closest match is retunred,
		whatever its distance. [default = -1]
	    
	Required for use with a BaseCatalog object.
	'''
	if len(self) == 0: return None
        NNindex,NNdist = self.NNindexdist(pos)
        if maxdist < 0. or maxdist >= NNdist:
            return NNindex
        else:
            return None
    
    def allwithin(self, pos, maxdist):
	'''Return a list of indices of all entries within maxdist of pos.
	
	This is high-level acess routine. 
	It returns a list of indices. If no object is found, the list is empty.
	    
	Required for use with a BaseCatalog object.
	'''
        dist=self.distto(pos)
        index= np.array(map(lambda d: maxdist >= d, dist)).nonzero()[0]
        return index