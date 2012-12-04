import numpy as np
#import warnings
import atpy

# TBD add support for non-scalars 

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
    object_null = None
    
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
                    #warnings.warn("WARNING: null value of column %s has been reset to default." % name)

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
        elif dtype == object:
            return self.object_null
        else:
            raise ValueError('No default null value for dtype %s' % dtype )
    
    def add_empty_column(self, name, dtype, **kwargs):
        if not 'null' in kwargs: kwargs['null'] = self.default_null(dtype)
        atpy.Table.add_empty_column(self, name, dtype, **kwargs)
        self.data[name][:] = self.columns[name].null
   
    def add_column(self, name, data, **kwargs):
        if not 'null' in kwargs: kwargs['null'] = self.default_null(data.dtype)
        atpy.Table.add_column(self, name, data, **kwargs)
    
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
        '''This method returns an empty row for the table, based on columns.null
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