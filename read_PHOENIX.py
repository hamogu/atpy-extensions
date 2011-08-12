# -*- coding: utf-8 -*-
import gzip
import atpy
import atpy.asciitables
import asciitable
import warnings
import re
import numpy as np
import scipy.interpolate

def read_block(file, lines_per_block):
    '''read one block of data

    PHOENIX often uses an fortran output, which print n values per line
    and breaks than the full array over several lines.
    The exponential notation used is 1.11D+01.
    read_block reads those block and joins them into a single numpy array of
    floating point values.
    
    input:

        file: file object from which the block is read

        lines_per_block: (int) Number of lines to to form a block
        This depends the number of layers used in PHOENIX and the 
        number of digtes output (e.g. more in .20 files than in .7 files
    ''' 
    block = []
    for i in range(lines_per_block):
          block.append(file.readline())
    block = ''.join(block).replace('\n','').replace('D','E')   # Exponential notation as 2.33D+03 -> 2.33E+03
    # in some formats negative numbers touch each other, the following regexp takes care of that
    return np.array(re.findall(r"([-]?\d\.\d*[eE][-+]\d+)",block), dtype='f') 


class exponentialDReader(asciitable.BasicReader):
    def __init__(self):
        asciitable.BasicReader.__init__(self)
        self.data.splitter.process_val = lambda x: x.replace('D','E').strip()

class Phoenix7Header(asciitable.BaseHeader):
    '''Hardcoded format for PHOENIX .7 files
    
    Usually this could be done with automatic splitting, but sometimes
    some of the integer columns for LineID (or whatever) get too big
    and join two field together.
    Although nobody cares for the later columns, this still causes
    problems with the auto-detection of columns. Thus, just hard-code the
    *interesting* columns here.
    '''
    def get_cols(self, lines):
        colstarts=[0,24,36,48,60,0]
        self.names = ['wave', 'flux', 'bbflux', 'flux2', 'otherstuff'] 
        self.cols = []
        for i in range(len(self.names)):
            col = asciitable.core.Column(name=self.names[i], index=i)
            col.start = colstarts[i]
            col.end = colstarts[i+1] - 1
            self.cols.append(col)

class Phoenix7Reader(asciitable.BasicReader):
    
    def __init__(self):
        asciitable.BasicReader.__init__(self)
        self.header = Phoenix7Header()
        self.data.splitter = asciitable.FixedWidthSplitter()
        self.data.splitter.process_val = lambda x: x.replace('D','E').strip()
        self.header.start_line = None
        self.data.start_line = 0



#def read_spectrum_gz(self, filename):
    #f = gzip.open(filename, 'rb')
    #file_content = f.read()
    #f.close()
    #read_spectrum(self, file_content)

def read_7(self, filename):
    atpy.asciitables.read_ascii(self, filename, Reader = Phoenix7Reader)
    self.keep_columns(['wave','flux','bbflux'])
    self.flux = 10.**self.flux
    self.bbflux = 10.**self.bbflux
    self.sort('wave')


# read tables from the .out files
# TBD: give table the name of the file opened (this requires checking if file is string = filename or a file object = file.name or a list -> name =''
def read_out(self, file, tableheader, iteration = -1):
    BaseInputter = asciitable.BaseInputter()
    lines = np.array(BaseInputter.get_lines(file))
    lines = np.char.strip(lines)
    
    # line looks like: 'no. of depth points: layer=           64'
    nlayers =  int(lines[(np.char.startswith(lines,'no. of depth points: layer='))][0].split(' ')[-1])
    
    #make list in which line the output for each iteration starts
    # find fist line with 'results of iteration no:' - usually this will be 1
    iter0 = int(lines[(np.char.startswith(lines,'results of iteration no:'))][0].split(' ')[-2])
    line_no_iter = np.char.startswith(lines,'results of iteration no:')
    temp = np.char.split(lines[line_no_iter], ' ').tolist()
    if len(temp) == 1:
        no_iter = np.array(temp[0][-2],dtype='int64')
    else:
        no_iter = np.array(np.array(temp[0:len(temp)-1])[:,-2],dtype='int64') # I have no idea why lists[0:9] works but lists itself does not
    line_no_iter = np.where(line_no_iter)[0]
    
    startcomp = line_no_iter[iteration]
    if iteration == -1:  # to the end of the file
        endcomp = len(lines)
    else:  #to the next iteration
        endcomp = line_no_iter[iteration + 1]
    
    tablestart = startcomp + np.where(np.char.startswith(lines[startcomp:endcomp],tableheader))[0]
    if len(tablestart) > 1:
        warnings.warn('Table header in ambiguous in iteration {!s} in table'.format(iteration))
    print 'This iterations is labelled: ', lines[line_no_iter[iteration]]
    print 'Using first table of'
    print lines[tablestart]
    tablestart = tablestart[0]
    atpy.asciitables.read_ascii(self, lines[tablestart:tablestart + nlayers + 1] , Reader=exponentialDReader)

### wrapper functions, which open the file if a string is passed and then pass the file object to the reader
def read_rfout(filename):
    if filename.endswith('.gz'):
        # support for with statement for gzip comes in Python 2.7
        #with gzip.open(args[0], mode) as file:
        #    function(self, file, args[1:], **kwargs)
        file = gzip.open(filename, 'r')
        try:

            return _read_rfout(file)
        finally:
            file.close()
    else:
        with open(filename) as file:
            return _read_rfout(file)

def _read_20(self, filename):
    if filename.endswith('.gz'):
        # support for with statement for gzip comes in Python 2.7
        #with gzip.open(args[0], mode) as file:
        #    function(self, file, args[1:], **kwargs)
        file = gzip.open(filename, 'r')
        try:

            return read_20(self, file)
        finally:
            file.close()
    else:
        with open(filename) as file:
            return read_20(self, file)

# read .7 files with rfout...
# This is a highly complicated format - It is much easier to write a dedicated reader then
#    to adapt an asciitable class to it
# in this simple version ignore atpy - I can put that in later

def _read_rfout(file):
    '''read PHOENIX .7 files which have been written with the rfout option
  
    These files contain an angular dependent radiation field at layer N
    of PHOENIX, which is set with rfout = N in the .dqs files.
    The angular dependence of the radiation field is interlaced with a normal 
    .7 file. For each wavelength there is one block of data, which contains the flux for 
    each angle.
    
    Input:
        file: file-like object
    '''
    
    print 'Reading file: Please be patient (rfout files are often several 100 MB)'
    flux1D = [file.readline().split()[1].replace('D','E')]
    n_ang = int(file.readline()) # number of angles
    # This formula rounds up, but the -1 makes is also work is n_ang is multiple of 10
    lines_per_block = (n_ang - 1) / 10 + 1 # (number angles -1) / (flux values per line = 10) +1
    #first block contain the angles of all rays in the form cos(mu), i.e. -1 <= cos(mu)<= +1
    angles = read_block(file, lines_per_block)
    wave = [file.readline().replace('D','E')]  # Exponential notation as 2.33D+03 -> 2.33E+03
    flux = [read_block(file, lines_per_block)]
    
    while True: 
        line = file.readline()   # the structure is 1) stuff, 2) wave, 3) block of flux values
        if line:                   # as long as stuff is not empty -> go on reading
          flux1D.append(line.split()[1].replace('D','E'))
          wave.append(file.readline())
          flux.append(read_block(file, lines_per_block))
        else:
            break
    
    flux1D = np.array(flux1D, dtype='f')
    wave = np.array(''.join(wave).replace('D','E').split(), dtype='f')
    flux = np.array(flux, dtype='f')
    index = wave.argsort()
    wave = wave[index]
    flux = flux[index,]
    flux1D = 10.**flux1D[index]
    return {'angles': angles, 'wave': wave, 'flux': flux, 'flux1D': flux1D}
    

def read_20(self, file):
    '''read PHOENIX .20 files 
    
    At the moment this only reads the upper part, not the NLTE information
    
    Input:
        self: atpy table object
        file: file-like object
    '''
    temp = file.readline() #discard first line
    n_layers = int(file.readline().split()[0])
    # This formula rounds up, but the -1 makes is also work is n_ang is multiple of 10
    lines_per_block = (n_layers - 1) / 3 + 1 # (number angles -1) / (flux values per line = 10) +1
    
    teff,rtau1,vtau1,pout,n,modtyp,identyp,vfold,rout = file.readline().replace('D','E').split()
    keys = {'teff': float(teff), 'rtau1': float(rtau1), 'vtau1': float(vtau1), 'pout': float(pout), 'n': int(n), 'modtyp': int(modtyp), 'identyp': int(identyp), 'vfold': float(vfold), 'rout': float(rout)}
    for key in keys:
        self.keywords[key] = keys[key]
    self.add_column('tstd', read_block(file, lines_per_block))
    self.add_column('temp', read_block(file, lines_per_block))
    while True:
        try:
            colname = file.readline()[0:13].strip().replace(' ','_')
            coldata = read_block(file, lines_per_block)
            self.add_column(colname, coldata)
        except ValueError:
            # reached end of block structure, one of these lines will haven an 'end-of-data' in them
            break

    
atpy.register_reader('.7',read_7, override = True)
atpy.register_extensions('.7',['7'])
atpy.register_reader('.out',read_out, override = True)
atpy.register_extensions('.out',['out'])
atpy.register_reader('.20',_read_20, override = True)
atpy.register_extensions('.20',['20', '20.gz'])  # _read_20 has its own mechanism to open gz files

#This compiles, now it has to be tested.
def rotation(wave, flux, v_rot, limb, resolution = 0.1, rotation_profile = None):
    '''applies rotational broadening to a spectrum

    This function is adapted from the IDL function rotten2
    (Ver. 1.0        SNS   GSFC   23/12/91)
    For calculating rotational broadening of line profiles using standard
    Uns"old-type profile and convolution.

    Input:
        wave: wave vector in Ang
        flux: flux vector
        v_rot: rotational velocity in km/s
        limb: coefficent of limb darkening
    Keywords:
        resolution = 0.1: wavelength resolution of output wave vector (in A)
    '''
    if len(wave) != len(flux):
        raise ValueError('wave and flux vector must have same length.')
    
    
    waveout = np.arange(np.min(wave), np.max(wave), resolution)
    ret = np.zeros(len(waveout),{'names':('wave','flux'), 'formats': (np.float32, np.float32)})
    ret['wave'] = waveout
    fluxinterpol = scipy.interpolate.interp1d(wave,flux)
    fluxout = fluxinterpol(ret['wave'])
    dlam = v_rot * ret['wave'] / 2.997e5
    q = 1.- (ret['wave']-np.mean(ret['wave']))**2./(dlam**2.)
    ind = (q >= 0.)
    q = q[ind]
    rotation_profile = 2./np.pi * q**0.5 + (limb/2.) * q
    rotation_profile = rotation_profile / (dlam[ind] * (1.+2. * limb/3.))
    rotation_profile = rotation_profile/np.sum(rotation_profile)
    ret['flux']= np.convolve(fluxout,rotation_profile,'same')
    return ret

#n= long((max(w)-min(w))/delta_lam)
#;print,n
#x=findgen(n)*delta_lam+min(w) 
#a_new=interpol(a,w,x)
#mean=(max(w)+min(w))/2.
#dlam=vrot*x/2.997e5
#q=1.-(x-mean)^2/(dlam^2)
#i=where(q ge 0.)
#q=q(i)
#rot=(2./3.1415926)*q^(0.5)+(limb/2.)*q
#rot=rot/(dlam*(1.+2.*limb/3.))
#rot=[0,rot,0]
#con=convol(a_new,rot/total(rot),center=1)
#j=where(con gt 0.)
#con=con(j)
#x=x(j)
#return
#end

#pro rotten2,w,a,delta_lam,limb,vrot,x,con,rot,a_new
#npar=n_params(0)
#if npar eq 0 then begin
#print,'rotation,w,a,delta_lam,limb,vrot,x,con,rot'
#print,'w = input wave vector'
#print,'a = input flux vector'
#print,'delta_lam = wavelength resolution of output wave vector (in A)'
#print,'limb = coefficent of limb darkening'
#print,'vrot = rotation velocity (kms)'
#print,'x = output wave vector'
#print,'con = output flux vector'
#print,'rot = output rotation profile'