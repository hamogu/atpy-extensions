from operator import itemgetter
import numpy as np
import copy
import warnings

import os

import asciitable
import Table     # which contains the Table which in turn inherits from atpy

import great_circle_dist

''' TBD
- If RA, DEC are not in table, error message is not very good.
- different names for RA, DEC? Sounds funny for cartesian coordinates. 
'''

try:
    import coords
    module_coords = True
except ImportError:
    module_coords = False

class SimpleCoords(object):
    '''A simple class to handle a column of coordinates for a catalog.
    
    The search is implemented in Python,
    thus its internals are readily available, which is great for debugging or
    customizing.
    
    There are two levels to acess the coordinates in a table:
        1) On the highly abstracted level routines return
           the index of the nearest neigbour or all sources within
           a given distance.
        2) On the lower level the coordinates themselves can be read out as
           SimpleCoords[2:4]. 
        3) On the lowest level Python allows access to all internals at own risk :-)

    '''
    def __init__(self, pos = np.array([],dtype = np.float64)):
        self.table = pos


    ### Convenience functions to make coordinates as easy to use as a Python sequence ###
    def __getitem__(self, item):
            if np.array(item).size == 0:
                return []
            #elif np.array(self.table).size == 1:
            #    return self.table
            else: return self.table[item]

    def __setitem__(self, key, value):
        if (key == 0) and (len(self.table) == 0): self.table= np.array([value], dtype = np.float64)
        else: self.table[key]=value

    def __len__(self):
        return len(self.table)

    def __getattr__(self, attribute):

        if attribute == 'RA':
            return self.table[:,0]
        elif attribute == 'DEC':
            return self.table[:,1]
        else:
            raise AttributeError(attribute)

    def set(self,pos):
        '''Replaces the full table of coordinates.

        Input:
            pos: N*2 np.ndarray for RA, DEC values in degrees
                 set will remove single dimensional entires and recast 2*N arrays 
        '''
        if ~isinstance(pos, np.ndarray): pos = np.array(pos)
        pos = np.squeeze(pos) #remove single dimensional entries
        if pos.ndim == 1:
            if (pos.shape == (0,)):
                self.table=np.array([],dtype = np.float64)
            elif (pos.shape == (2,)):
                self.table = pos.reshape(1,2)
            else:
                raise ValueError('Input coordinate table must be of shape (N, 2)')
        elif (pos.ndim == 2) & (pos.shape[1] == 2):
            self.table = pos
        else:
            raise ValueError('Input coordinate table must be of shape (N, 2)')

    def __str__(self):
        str(self.table)


    ### Functions to change set of coordinates ###
    def calc(self, pos, units=None):
        '''Do nothing. This is a hook for more complicated classes.

        pos: coordinates'''
        if units: warnings.warn('SimpleCoords do not support automatic unit conversion - Ignored!')
        return pos

    def reset(self):
        '''Reset the table to empty, but keep all other attributes.'''
        self.table=np.array([],dtype = np.float64)

    def append(self, pos):
        '''Append a np.ndarray        '''
        if (pos.ndim ==2 and pos.shape[1] == 2) or (pos.ndim ==1 and pos.shape[0] == 2):
            self.table=np.vstack((self.table, pos))
        else:
            raise ValueError('Input coordinate table must be of shape (N,2) or (2,)')
        


    ### Functions to search in the coordinates ###
    def distto(self,pos, ind= slice(None)):
        '''Calculate an array of distances of all elements to position pos

        This ia medium-level acess routine. 
        It return a list of coords.AngSep objects.
        input:
           pos: [RA, DEC] in deg of footpoint for distance
        keyword:
           ind: set to index array to calculate distances only for a subset
               of all coordinates (default = `slice(None)` -> all)
        '''
        return great_circle_dist.dist_radec(pos[0], pos[1], self.table[ind,0], self.table[ind,1], unit='deg')

    def NNindexdist(self, pos):
        '''Return index and distance of the entry closest to pos.

        This ia medium-level acess routine. 
        It returns an index and a distance.
        input:
            pos: [RA, DEC] in deg of footpoint for distance
        '''
        dist=self.distto(pos)
        return np.argmin(dist), np.min(dist)

    def NNindex(self, pos, maxdist=-1., units = None):
        '''Return index of the entry closest to pos.

        This is high-level acess routine. 
        It returns an index.
        input:
            pos: [RA, DEC] in deg of footpoint for distance
        keywords:
            maxdist: If no match within maxdist [in deg] is found, None is returned.
                For negative values of maxdist the closest match is returned,
                whatever its distance. [default = -1]
        '''
        if units: warnings.warn('Coords to not support automatic unit conversion - Ignored!')
        if len(self) == 0: return None
        NNindex, NNdist = self.NNindexdist(pos)
        if maxdist < 0. or maxdist >= NNdist:
            return NNindex
        else:
            return None

    def allwithin(self, pos, maxdist, units = None):
        '''Return a list of indices of all entries within maxdist of pos.

        This is high-level acess routine. 
        It returns a list of indices. If no object is found, the list is empty.
        input:
            pos: [RA, DEC] in deg of footpoint for distance
            maxdist: If no match within maxdist is found, None is returned.
                For negative values of maxdist the closest match is returned,
                whatever its distance. [default = -1]
        '''
        if units: warnings.warn('Coords to not support automatic unit conversion - Ignored!')
        dist=self.distto(pos)
        return (dist <= maxdist).nonzero()[0]

if module_coords:
  class CoordsClassCoords(object):
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

    SimpleCoords is good for Catalogs with < 1000 entries for situations
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
    return zip(data[self._ra],data[self._dec])

def coord_RADECstring(self, data):
    return map(lambda line: str(line[0])+' '+str(line[1]),zip(data[self._ra], data[self._dec]))

def coord_base(self, data):
    raise NotImplementedError

def coord_simple(self, data):
    '''form N*2 array from data[self._ra] and data[self._dec]'''
    return np.array([data[self._ra], data[self._dec]]).transpose()

def coord_str2RADEC(self, data):
    t = asciitable.read(data[self._ra], delimiter=':', Reader = asciitable.NoHeader, names =  ['h','m','s'])
    RA = t.h*15. + t.m / 4. + t.s/4./60.
    t = asciitable.read(data[self._dec], delimiter=':', Reader = asciitable.NoHeader, names =  ['d','m','s'])
    DEC = t.d + t.m / 60. + t.s/3600.
    return coord_simple(self, {self._ra: RA, self._dec: DEC})

class BaseCatalog(Table.Table):
    
    def __init__(self, *args, **kwargs):
        self._ra = 'RA'
        self._dec = 'DEC'
        if 'form_coords' in kwargs: self.form_coords = (kwargs.pop('form_coords'))
        else: self.form_coords = coord_base
        if 'Coords' in kwargs: self.coords = (kwargs.pop('Coords'))
        else: self.coords = SimpleCoords()
        Table.Table.__init__(self, *args, **kwargs)
        self.match_source = match_source_dist
        self.match_source_dist = 1./3600.

    
    def calc_all_coords(self):
        self.coords.set(self.coords.calc(self.form_coords(self, self.data)))
    ### Extension of ATPy functions which effect self.coords
    
    def read(self, *args, **kwargs):
        if 'RA' in kwargs:
            self._ra = kwargs.pop('RA')
        if 'DEC' in kwargs:
            self._dec = kwargs.pop('DEC')
        if 'form_coords' in kwargs:
            self.form_coords = kwargs.pop('form_coords')
        Table.Table.read(self, *args, **kwargs)
        self.calc_all_coords()
    
    def add_column(self, name, data,  **kwargs):
        '''calc coords if second column of RA DEC has been added
        
        This prescription only covers the standard case of columns with self._ra and self._dec
        If self.form_coords is overridden by a more complex prescrition calc_all_coords()
        may have to be called explicitly by the user.
        '''
        Table.Table.add_column(self, name, data,  **kwargs)

        if name in (self._ra, self._dec) and self._ra in self.names and self._dec in self.names and \
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
        Table.Table.reset(self)
        self.coords.reset()
    
    def append(self, newdata, pos=[]):
        Table.Table.append(self, newdata)
        if len(pos) > 0:
            self.coords.append(pos)
        else:
            self.coords.append(map(self.coords.calc,self.form_coords(self, newdata)))
    
    
    ### direct extension of ATPy functions which require adding self.coords
    def where(self, mask):
        new_table = Table.Table.where(self, mask)
        new_table.coords = copy.copy(self.coords)
        if self.coords: new_table.coords.set(self.coords[mask]) #set, if list is not empty
        new_table._ra = self._ra
        new_table._dec = self._dec
        new_table.form_coords = self.form_coords
        new_table.match_source = self.match_source
        return new_table
    
    def row(self, row_number, python_types = False):
        row = Table.Table.row(self, row_number, python_types = python_types)
        return row, self.coords[row_number]
    
    
    ### indirect extenstion of ATPy - functions defined in Table
    def empty_row(self):
        emptyrow = Table.Table.empty_row(self)
        if emptyrow: emptyrow.coords.reset()
        return emptyrow
    
    def update_row(self, rowindex, data, auto_add_columns = False, overwrite = False):
        Table.Table.update_row(self, rowindex, data, auto_add_columns = auto_add_columns, overwrite = overwrite)
        #update coordinates is necessary
        if overwrite:
            if overwrite == True: overwritenames = set(self._fields(data)) #overwrite = True
            else: overwritenames = set(overwrite)  #overwrite is a list of columnnames
        else: overwritenames = set([ ])
        if (self._ra in overwritenames) or (self._dec in overwritenames):
            self.coords[rowindex] = self.coords.calc(self.form_coords(self, self.rows([rowindex]))[0])
    
    def add_row(self, data, pos, auto_add_columns = False):  #Do I really need to copy everything, just to get the pos=pos in there? 
        ''' If pos is given, it needs to be of coord type, if not use coords.calc first
        '''
        # The following line works because of short-circuting - NoneType has no len()!
        if  isinstance(self.data, type(None)) or len(self.data) == 0:
                add_first_row_in_table = True
                if not (self._ra in self._fields(data) and self._dec in self._fields(data)): raise Exception('First row added to the table must contain RA and DEC')
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
            #self.coords[0] = self.coords.calc(self.form_coords(self, data))
            self.update_row(0, newrow, overwrite = True)
            self.coords[0] = pos
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
            self.add_row(data, pos, auto_add_columns = auto_add_columns)
        else:
            self.update_row(match_source, data, auto_add_columns = auto_add_columns, overwrite = overwrite)
    
    def add_catalog(self, pos, data, auto_add_columns = False, overwrite = False):
        for i in range(len(data)):
            self.add_source_info(pos[i], data[[i]], auto_add_columns = auto_add_columns, overwrite = overwrite)
    
    def update_rows(self, pos, data, auto_add_columns = False, overwrite = False): #, subset = slice(None)):
        '''Update rows in a catalog with information from a different catalog
        
        Contrary to add_catalog, update_rows does not add sources to the catalog, but only adds
        information to existing sources, e.g. add te JHK magnitudes from 2MASS to an X-ray catalog,
        but do not add those 2MASS sources without X-ray counterpart.
        
        The procedures searched for matching sources with self.match_source and fills empty fields
        in those sources which are identified as matches with information from data.
        
        TBD: This methods will allow to pass an index array to operate only on a subset of the original catalog.
        
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
        #if subset.dtype == 'bool':
            #subsetind, = subset.nonzero()
        #elif index == None:
            #subsetind = range(len(self.data))
        #else:
            #subsetind = index
        #subsettable = self.where(subset)
        for i in range(len(data)):
            match_source=self.match_source(self, pos[i],data[i])
            if match_source != None:
                self.update_row(match_source, data[i], auto_add_columns = auto_add_columns, overwrite = overwrite)
    

def match_source_dist(self, pos, data):
   return self.coords.NNindex(pos, maxdist = self.match_source_dist)

def match_source_name(self, pos, data):
        return np.where(self['name'] == pos)[0]

def match_source_none(self,pos,data):
    return None

class SimpleCatalog(BaseCatalog):
    '''A BaseCatalog initialized with SimpleCoords as coordinate table 
    '''
    def __init__(self, *args, **kwargs):
        kwargs['Coords'] = SimpleCoords()
        if not 'form_coords' in kwargs:
            kwargs['form_coords'] = coord_simple
        BaseCatalog.__init__(self, *args, **kwargs)

class CartesianCatalog(BaseCatalog):
    '''A BaseCatalog initialized with CarteseanCoords as coordinate table 
    '''
    def __init__(self, *args, **kwargs):
        kwargs['Coords'] = CartesianCoords()
        if not 'form_coords' in kwargs:
            kwargs['form_coords'] = coord_simple
        BaseCatalog.__init__(self, *args, **kwargs)

class CoordsClassCatalog(BaseCatalog):
    '''A BaseCatalog initialized with CoordsClassCoords as coordinate table 
    '''
    def __init__(self, *args, **kwargs):
        kwargs['Coords'] = SimpleCoords()
        BaseCatalog.__init__(self, *args, **kwargs)
        self.form_coords = coord_RADEC
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

class StetsonStandardCatalog(BaseCatalog):


        def __init__(self, *args, **kwargs):
                if len(args) == 0: BaseCatalog.__init__(self)
                else:
                        # no arg required in code, but need to put any dummy
                        # because a call with no argument would create an empty dummy
                        posfile = os.path.join(os.path.dirname(__file__), '..', 'catalog', 'Land.pos')
                        magfile = os.path.join(os.path.dirname(__file__), '..', 'catalog', 'hans.dat')
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
                        mags = Table.Table(magfile, type = 'ascii', data_start = 1, Reader = asciitable.NoHeader, names = names, include_names = include_names)
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
        '''Do nothing. This is just a hook for more complicated Coords.
        '''
        return pos
    
    
    ### Functions to search in the coordinates ###
    def distto(self,pos):
        '''Calculate an array of distances of all elements to position pos
        
        This is medium-level acess routine. 
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
