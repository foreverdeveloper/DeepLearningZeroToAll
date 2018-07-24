'''
https://www.python-course.eu/numpy_reading_writing.php

    savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ')

    Parameter	Meaning
    X	:   array_like Data to be saved to a text file.
            str or sequence of strs, optional
            A single format (%10.5f), a sequence of formats, or a multi-format string,
                e.g. 'Iteration %d -- %10.5f',
            in which case 'delimiter' is ignored. For complex 'X', the legal options for 'fmt' are:
    fmt	:
        a) a single specifier, "fmt='%.4e'", resulting in numbers formatted like "' (%s+%sj)' % (fmt, fmt)"
        b) a full string specifying every real and imaginary part,
            e.g. "' %.4e %+.4j %.4e %+.4j %.4e %+.4j'" for 3 columns
        c) a list of specifiers, one per column - in this case, the real and imaginary part must have separate specifiers,
            e.g. "['%.3e + %.3ej', '(%.15e%+.15ej)']" for 2 columns
    delimiter   : A string used for separating the columns.
    newline	    : A string (e.g. "\n", "\r\n" or ",\n") which will end a line instead of the default line ending
    header	    : A String that will be written at the beginning of the file.
    footer	    : A String that will be written at the end of the file.
    comments	: A String that will be prepended to the 'header' and 'footer' strings, to mark them as comments.
                The hash tag '#' is used as the default.

'''

import numpy as np
x = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]], np.int32)
np.savetxt("test.txt",  x)  # 출력 1
np.savetxt("test2.txt", x, fmt="%2.3f"  , delimiter=",") # 출력 2
np.savetxt("test3.txt", x, fmt="%04d"   , delimiter=" :-) ")  # 출력 3
'''
출력
-----------------------------------------------------------------------------
1.000000000000000000e+00 2.000000000000000000e+00 3.000000000000000000e+00
4.000000000000000000e+00 5.000000000000000000e+00 6.000000000000000000e+00
7.000000000000000000e+00 8.000000000000000000e+00 9.000000000000000000e+00
-----------------------------------------------------------------------------//

출력 2
-----------------------------------------------------------------------------
1.000,2.000,3.000
4.000,5.000,6.000
7.000,8.000,9.000
-----------------------------------------------------------------------------//

출력 3
-----------------------------------------------------------------------------
0001 :-) 0002 :-) 0003
0004 :-) 0005 :-) 0006
0007 :-) 0008 :-) 0009
-----------------------------------------------------------------------------//

'''



print("------------------------Loading Textfiles with loadtxt -------------------------------")
y = np.loadtxt("test.txt")
print(y)
'''
출력
---------------------------------------
[[1. 2. 3.]
 [4. 5. 6.]
 [7. 8. 9.]]
---------------------------------------//
'''

y = np.loadtxt("test2.txt", delimiter=",")
print(y)
'''
출력
---------------------------------------
[[1. 2. 3.]
 [4. 5. 6.]
 [7. 8. 9.]]
---------------------------------------//
'''

y = np.loadtxt("test3.txt", delimiter=" :-) ")
print(y)
'''
출력
---------------------------------------
[[1. 2. 3.]
 [4. 5. 6.]
 [7. 8. 9.]]
---------------------------------------//
'''

print("It's also possible to choose the columns by index:")
y = np.loadtxt("test3.txt", delimiter=" :-) ", usecols=(0,2))   # 컬럼 0, 1
print(y)
'''
출력
---------------------------------------
[[1. 3.]
 [4. 6.]
 [7. 9.]]
---------------------------------------//
'''

# We define first a function which converts "hh::mm::ss" into minutes:
def time2float_minutes(time):
    if type(time) == bytes:
        time = time.decode()
    t = time.split(":")
    minutes = float(t[0])*60 + float(t[1]) + float(t[2]) * 0.05 / 3
    return minutes

for t in ["06:00:10", "06:27:45", "12:59:59"]:
    print(time2float_minutes(t))
'''
출력
---------------------------------------
360.1666666666667
387.75
779.9833333333333
---------------------------------------//
'''


'''
Write a program, using the newly written generator "trange", to create a file "times_and_temperatures.txt". 
The lines of this file contain a time in the format hh::mm::ss and random temperatures between 10.0 and 25.0 degrees. 
The times should be ascending in steps of 90 seconds starting with 6:00:00.

You might have noticed that we check the type of time for binary. 
The reason for this is the use of our function "time2float_minutes in loadtxt in the following example. 
The keyword parameter converters contains a dictionary which can hold a function for a column 
(the key of the column corresponds to the key of the dictionary) to convert the string data of this column into a float. 
The string data is a byte string. That is why we had to transfer it into a a unicode string in our function:

 
For example: times_and_temperatures.txt
----------------------------------------
06:00:00 20.1
06:01:30 16.1
06:03:00 16.9
06:04:30 13.4
06:06:00 23.7
06:07:30 23.6
06:09:00 17.5
06:10:30 11.0
----------------------------------------//
'''
y = np.loadtxt("times_and_temperatures.txt",
               converters={ 0: time2float_minutes})
print(y)

'''
출력
------------------------------
[[360.   20.1]
 [361.5  16.1]
 [363.   16.9]
 [364.5  13.4]
 [366.   23.7]
 [367.5  23.6]
 [369.   17.5]
 [370.5  11. ]]
------------------------------//
# delimiter = ";" , # i.e. use ";" as delimiter instead of whitespace 


'''

print('---------------------------------- tofile ---------------------------------------')
'''
tofile

    tofile is a function to write the content of an array to a file both in binary, 
    which is the default, and text format.
    
        A.tofile(fid, sep="", format="%s")
        
    The data of the A ndarry is always written in 'C' order, regardless of the order of A.
    The data file written by this method can be reloaded with the function fromfile().
    

    ------------------------------------------------------------------------------------------------
    Parameter	Meaning
    ------------------------------------------------------------------------------------------------
    fid	: can be either an open file object, or a string containing a filename.
    sep	: The string 'sep' defines the separator between array items for text output. 
            If it is empty (''), a binary file is written, equivalent to file.write(a.tostring()).
            
    format	Format string for text file output. 
            Each entry in the array is formatted to text by first converting it to the closest Python type, 
            and then using 'format' % item.
    ------------------------------------------------------------------------------------------------//        

    Remark:
        Information on endianness and precision is lost. 
        Therefore it may not be a good idea to use the function to archive data or transport data between machines with 
        different endianness. Some of these problems can be overcome by outputting the data as text files, 
        at the expense of speed and file size.
     
'''
dt = np.dtype([('time', [('min', int), ('sec', int)]),
               ('temp', float)])
x = np.zeros((1,), dtype=dt)
x['time']['min'] = 10
x['temp'] = 98.25
print(x)    # [((10, 0), 98.25)]
fh = open("test6.txt", "bw")
x.tofile(fh)


print('---------------------------------- fromfile ---------------------------------------//')
'''
fromfile to read in data, which has been written with the tofile function. 
It's possible to read binary data, if the data type is known. 
It's also possible to parse simply formatted text files. The data from the file is turned into an array.

The general syntax looks like this:
    numpy.fromfile(file, dtype=float, count=-1, sep='')
    
Parameter	Meaning
    file    : 'file' can be either a file object or the name of the file to read.
    dtype   : defines the data type of the array, which will be constructed from the file data. 
                For binary files, it is used to determine the size and byte-order of the items in the file.
    count   : defines the number of items, which will be read. -1 means all items will be read.
    sep     : The string 'sep' defines the separator between the items, if the file is a text file. 
                If it is empty (''), the file will be treated as a binary file. 
                A space (" ") in a separator matches zero or more whitespace characters. 
                A separator consisting solely of spaces has to match at least one whitespace.    

'''
fh = open("test6.txt", "rb")
x = np.fromfile(fh, dtype=dt)
print(x)

print('---------------------------------- to/fromfile ---------------------------------------//')
'''
Attention:

    It can cause problems to use tofile and fromfile for data storage, 
    because the binary files generated are not platform independent. 
    There is no byte-order or data-type information saved by tofile. 
    Data can be stored in the platform independent .npy format using save and load instead.

'''
import numpy as np
import os
# platform dependent: difference between Linux and Windows
#data = np.arange(50, dtype=np.int)
data = np.arange(50, dtype=np.int32)
data.tofile("test4.txt")
fh = open("test4.txt", "rb")
# 4 * 32 = 128
fh.seek(128, os.SEEK_SET)
x = np.fromfile(fh, dtype=np.int32)
print(x)
'''
출력
--------------------------------------------------------------------
[32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49]
--------------------------------------------------------------------//
'''


print('=============================== Best Practice to Load and Save Data ===================================')
'''
The recommended way to store and load data with Numpy in Python consists in using load and save. 
We also use a temporary file in the following :

'''
import numpy as np
from tempfile import TemporaryFile
outfile = TemporaryFile()
x = np.arange(10)
print(x)
np.save(outfile, x)
outfile.seek(0) # Only needed here to simulate closing & reopening file
np.load(outfile)
'''
출력
--------------------------------------------------------------------
[0 1 2 3 4 5 6 7 8 9]
--------------------------------------------------------------------//
'''

print('=============================== and yet another way: genfromtxt ===================================')
'''
There is yet another way to read tabular input from file to create arrays. 
As the name implies, the input file is supposed to be a text file. 
The text file can be in the form of an archive file as well. 
genfromtxt can process the archive formats gzip and bzip2. 
The type of the archive is determined by the extension of the file, i.e. '.gz' for gzip and bz2' for an bzip2.

genfromtxt is slower than loadtxt, but it is capable of coping with missing data. 
It processes the file data in two passes. 
At first it converts the lines of the file into strings. 
Thereupon it converts the strings into the requested data type. loadtxt on the other hand works in one go, 
which is the reason, why it is faster.



'''

print('=============================== recfromcsv(fname, **kwargs) ===================================')
'''
This is not really another way to read in csv data. 'recfromcsv' basically a shortcut for
    np.genfromtxt(filename, delimiter=",", dtype=None)

'''
