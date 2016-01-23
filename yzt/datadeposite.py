import sys
import types

# The function of this file is to initial attributes
# In this file,it defines several classes,which will be used in other place

class DataDeposite(object):
# This class's main function is to initial variables in __dict__

# _attrs is an attribute deposite,every attribute we want to add will 
# firstly add into this varibles.
# _nodump_attrs is a temporal attribute deposite,it can only save some 
# temporal attributes which may brush off finally
    _attrs = []
    _nodump_attrs = []
# init DataDeposite.First update attributes and values in this deposite.
# If attributes were not in __dict__,but in _attrs and _nodump_attrs,add it,
# and assign it to None
    def __init__(self, **entries):
        self.__dict__.update(entries)
        for key in (self._attrs + self._nodump_attrs):
            if key not in self.__dict__:
                self.__dict__.update({key:None})
# refresh keys in self.__dict__(attention:that's not means change attributes 
# in __dict__,it firstly copy attributes in __dict__ to odict,then refresh keys
# in odict)
# if keys in odict are no longer in _attrs,delete it 
    def __getstate__(self):
        odict = self.__dict__.copy()
        for key in odict.keys():
            if key not in self._attrs:
                del odict[key]
        return odict 
# Redirect __repr__,show attributes and values in _attrs and _nodump_attrs
    def __repr__(self):
        args = [ '%s=%s' % (k, vars(self).get(k)) for k in self._attrs + self._nodump_attrs]
        return 'DataDeposite(%s)' % ', '.join(args)


class DataDepositeTk(object):
# Toolkit for DataDeposite,including some operations

# Create attribute depo for saving varibles in depo
# if no varibles in depo,run depocls to add varibles
    def __init__(self, depo=None, depocls=None, *args, **kargs):
        if depo is not None:
            self.depo = depo
        else:
            self.depo = depocls(*args, **kargs)

# return depo
    def getdepo(self):
        return self.depo
# set depo into self.depo     
    def setdepo(self, depo):
        self.depo = depo
# transform attributes and values from self.depo into __dict__? I guess
    def deposite(self, value, name, cache=True):
        if cache:
            setattr(self.depo, name, value)
# clear attributes in __dict__ and _nodump_attrs,delete these keys
    def clear_nodump(self):
        for key in self.depo._nodump_attrs:
            if key in self.depo.__dict__:
                del self.depo.__dict__[key]

class DataDepositeDecorator(DataDepositeTk):
# Decorate deposite 

# add attributes and values into self.depo
    def add_attr(self, name, value):
        setattr(self.depo, name, value)
# redirect __dir__,return attributes in __dict__ and in depo.__dict__
    def __dir__(self):
        return self.__dict__.keys() + self.depo.__dict__.keys()
# get attributes from self.depo
    def __getattr__(self, key):
        if hasattr(self.depo, key):
            return getattr(self.depo, key)
        else:
            raise AttributeError, key
# screen data type
def DataTypedef(cls, attrs):
    return type(cls, (DataDeposite,), {'_attrs':attrs})


class Struct:
# define a class named Struct,with method of initialization and representation

# init Struct class
    def __init__(self, **entries):
        self.__dict__.update(entries)
# screen attributes
    def __repr__(self):
        args = [ '%s=%s' % (k, repr(v)) for (k,v) in vars(self).items()]
        return 'Struct(%s)' % ', '.join(args)

# Load modules into memory,and update attributes in struct into __main__
def dump_struct_to_main(struct):
    sys.modules['__main__'].__dict__.update(struct.__dict__)
# Update attributes and values
def update(x, **entries):
    if type(x) == types.DictType:
        x.update(entries)
    else:
        x.__dict__.update(entries)
    return x
