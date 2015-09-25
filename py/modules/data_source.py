#!/usr/bin/env python

''' Data source for SSA '''

import numpy as np
#import ConfigParser # errors
import re
import logging
import config

class DataSrcError(Exception):
    pass

class IDataSrc(object):
    def __init__(self, cn, src_cnf, cnf):
        self._class_name = cn
        self._cnf = cnf
        self._src_cnf = src_cnf

    def make(self):
        raise DataSrcError("pure virtual 'make()' in %s" % self._class_name)

    def configure(self):
        raise DataSrcError("pure virtual 'cnf()' in %s" % self._class_name)

    def name(self):
        return self._class_name

    def data_size(self):
        return self._cnf.common.data_size


class CsvSrc(IDataSrc):
    def __init__(self, src_cnf, cnf):
        IDataSrc.__init__(self, "CsvSrc", src_cnf, cnf)
        self._file = 0
        self._data_sz = 0           # min(_data sz, _common_data_sz)
        self._csv_file = ""
        self._data = None
        self._log = logging.getLogger("CsvSrc")
        self._cols = []
        self._use = ''

    def data_size(self):
        ''' reimplemented virtual method '''
        return self._data_sz

    def make(self):
        return self._data.copy()

    def configure(self):
        try:
            if self._csv_file == self._src_cnf.filename and \
                    self._src_cnf.cols == self._cols and \
                    self._use == self._src_cnf.use:
                return
            self._csv_file = self._src_cnf.filename
            self._cols = self._src_cnf.cols
            self._use = self._src_cnf.use
            if len(self._cols) == 0:
                raise DataSrcError("%s: no one col names in config"
                        % (self._class_name))
            # read data from csv
            csv_data = np.recfromcsv(self._csv_file, autostrip=True,
                    dtype=float, usecols=self._cols)
            self._data_sz = csv_data.size
            self._log.info("data file: %s" % self._csv_file)
            self._log.info("data size: %d" % self.data_size())
            self._log.info("self._cols: %s" % self._cols)
            self._log.info("self._use: %s" % self._use)
            # process readed data
            self._data = np.zeros(csv_data.size)
            if self._use == 'middle':
                for col in self._cols:
                    self._data += csv_data[col]
                self._data /= len(self._cols)
            elif self._use == 'delta':
                self._data = csv_data[self._cols[0]]
                for idx in range(1, len(self._cols)):
                    self._data -= csv_data[self._cols[idx]]
            else:
                # this is a column name
                self._data = csv_data[self._cols[0]]
        except config.ConfigError, e:
            err = "%s: %s" % (self._class_name, repr(e))
            self._log.error(err)
            raise DataSrcError(err)


class LinSrc(IDataSrc):
    def __init__(self, src_cnf, cnf):
        IDataSrc.__init__(self, "LinSrc", src_cnf, cnf)
        self._y0 = 0.0
        self._angle = 0.0
        self._log = logging.getLogger("LinSrc")
        self._regenerate = True
        self._y = None

    def make(self):
        if not self._regenerate:
            self._log.debug("return previous data(_regenerate)")
            return self._y.copy()
        self._log.debug("recalculate")
        x = np.array(range(self.data_size()))
        tan = np.tan(np.pi / 180.0 * self._angle)
        self._y = self._y0 + tan * x
        self._regenerate = False
        return self._y.copy()   # important. Without copy() reference
                                # returns and operation on data will
                                # corrupt saved here values.

    def configure(self):
        try:
            if self._y0 == self._src_cnf.y0 and \
                    self._angle == self._src_cnf.angle:
                return
            self._y0 = self._src_cnf.y0
            self._angle = self._src_cnf.angle
            self._regenerate = True
        except config.ConfigError, e:
            err = "%s: %s" % (self._class_name, repr(e))
            self._log.error(err)
            raise DataSrcError(err)


class SinSrc(IDataSrc):
    def __init__(self, src_cnf, cnf):
        IDataSrc.__init__(self, "SinSrc", src_cnf, cnf)
        self._amp = 0.0
        self._period = 0
        self._phase = 0.0
        self._log = logging.getLogger("LinSrc")
        self._regenerate = True
        self._y = None

    def make(self):
        if not self._regenerate:
            return self._y.copy()
        if self._amp == 0 or self._period == 0 or self.data_size() == 0:
            raise DataSrcError("SinSrc: some settings is zero!")
        x = np.ones(self.data_size())
        for i in range(x.size):
            x[i] = 1.0 * i / self._period * 2 * np.pi
        x = np.remainder(x, 2 * np.pi)
        self._y = self._amp * np.sin(x)
        self._regenerate = False
        return self._y.copy()

    def configure(self):
        try:
            if self._amp == self._src_cnf.amp and \
                    self._period == self._src_cnf.period and \
                    self._phase == self._src_cnf.phase:
                return
            self._period = self._src_cnf.period
            self._amp = self._src_cnf.amp
            self._phase = self._src_cnf.phase
            self._log.info("period:{:f}, amp:{:f}, phase:{:f}, data size: {:d}"
                    .format(self._period, self._amp, self._phase, self.data_size()))
            self._regenerate = True
        except config.ConfigError, e:
            err = "%s: %s" % (self._class_name, repr(e))
            self._log.error(err)
            raise DataSrcError(err)


class RandSrc(IDataSrc):
    ''' return data vector have length N'''
    def __init__(self, src_cnf, cnf):
        IDataSrc.__init__(self, "RandSrc", src_cnf, cnf)
        self._log = logging.getLogger("RandSrc")
        self._distr = 'unknonwn'
        self._lim_min = 0.0
        self._lim_max = 0.0
        self._use_cached = False
        self._y = None

    def make(self):
        if not self._regenerate and self._use_cached and not self._y is None:
            self._log.debug("return previous data(_regenerate or _use_cached)")
            return self._y.copy()
        self._log.debug("recalculate")
        if self._src_cnf.distr_type == 'uniform':
            self._regenerate = False
            self._y = np.random.uniform(self._lim_min, self._lim_max, self.data_size())
            return self._y.copy()
        self._log.error("type {} not implemented.".format(
            self._src_cnf.distr_type))
        return np.zeros(self.data_size)

    def configure(self):
        if self._distr == self._src_cnf.distr_type and \
                    self._lim_min == self._src_cnf.lim_min and \
                    self._lim_max == self._src_cnf.lim_max and \
                    self._use_cached == self._src_cnf.use_cached:
            return
        self._distr = self._src_cnf.distr_type
        self._lim_min = self._src_cnf.lim_min
        self._lim_max = self._src_cnf.lim_max
        self._use_cached = self._src_cnf.use_cached
        self._regenerate = True


# Note: no spaces allowed between [] and section name in cnf file!
class DataSource(object):
    def __init__(self):
        self._src_list = []
        self._cnf = None
        self._data_sz = 0
        self._log = logging.getLogger("DataSrc")

    def append_src(self, src_cnf, cnf):
        global sources
        stype = src_cnf.TYPE
        self._cnf = cnf
        if not stype in sources.keys():
            raise DataSrcError("Config error, unknown source '%s'"
                    % (stype))
        # append new object of concrete source
        obj = sources[stype](src_cnf, cnf)
        obj.configure()  # first time. Really checking configuration data.
        self._src_list.append(obj)

    def make(self):
        res = None
        self._data_sz = 0
        self._log.debug("Data producing:")
        for s in self._src_list:
            self._log.debug("src {}".format(s.name()))
            s.configure()
            y = s.make()
            self._log.debug("s.make() res({}): {}".format(s.name(), y))
            if not self._data_sz:
                res = y
                self._data_sz = y.size
            else:
                if res.size != y.size:
                    msg = "DataSource: differ data size of sources!"
                    msg += " current: %d, from %s: %d" % (res.size, y.size, s.name())
                    raise DataSrcError(msg)
                res += y
        return res

    def data_size(self):
        return self._data_sz

sources = { config.LinearSourceConfig.TYPE : LinSrc,
        config.SinSourceConfig.TYPE : SinSrc,
        config.CSVSourceConfig.TYPE : CsvSrc,
        config.RandomSourceConfig.TYPE: RandSrc}

def create_data_source(cnf):
    ''' Create data source by config. Config is an instance of
        ConfigParser class.
        It should be section 'source.N' for concrete source type.
        Content of that source is specific.
    '''
    log = logging.getLogger("CreDataSource")
    global sources
    log.info("create new DataSource")
    data_source = DataSource()
    try:
        if cnf.common.source == 'combined':
            for src_cnf in cnf.sources:
                data_source.append_src(src_cnf, cnf)
                log.info("append {}".format(src_cnf.sname))
        else:
            print "create_data_sources: src_idx:", cnf.source_idx
            src_cnf = cnf.sources[cnf.source_idx] # concrete config
            data_source.append_src(src_cnf, cnf)
            log.info("append {}".format(src_cnf.sname))
    except config.ConfigError, e:
        raise DataSrcError("Config error: %s" % (repr(e)))
    return data_source

if __name__ == '_main__':
    import ConfigParser
    cnf = ConfigParser.ConfigParser()
    cnf.add_section('source.1')
    cnf.set('source.1', 'type', 'sin')
    cnf.set('source.1', 'period', 10.)
    cnf.set('source.1', 'amp', 1.5)

    cnf.add_section('source.common')
    cnf.set('source.common', 'data_size', 25)
    create_data_source(cnf)

