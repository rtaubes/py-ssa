#!/usr/bin/env python
'''
    1) Make SSA analyze of input,
    2) create SSA distribution,
    3) create data parts from SSA as components
    of the original data
'''
import logging
import configparser
import re


class ConfigError(Exception):
    pass


class CommonConfig(object):
    help = ['section "source.common".',
            '  parameters:',
            '   - data_size - how much data should be got from source',
            '   - source - name of data source as "source.N" or "combined" for summ of them',
            '   - predict - length of prediction interval',
            '   - data_offset - how much skip before analyse']
    SNAME = 'source.common'
    def __init__(self, sname, conf_obj):
        self.data_size = conf_obj.getint(self.SNAME, 'data_size')
        # current source, used in application
        self.source = conf_obj.get(self.SNAME, 'source')
        self.predict = conf_obj.getint(self.SNAME, 'predict')
        self.data_offset = conf_obj.getint(self.SNAME, 'data_offset')

    def save(self, conf_obj):
        conf_obj.add_section(self.SNAME)
        conf_obj.set(self.SNAME, 'data_size', self.data_size)
        conf_obj.set(self.SNAME, 'source', self.source)
        conf_obj.set(self.SNAME, 'predict', self.predict)
        conf_obj.set(self.SNAME, 'data_offset', self.data_offset)


class CSVSourceConfig(object):
    help = ['section source.N, "csv".',
            '  parameters:',
            '   - type: "csv"',
            '   - file: csv file name',
            '   - use: (high/low/open/close/vol/average) use in calculation',
            '   - cols - which colums use calculation of average']

    TYPE = 'csv'
    ID = 0
    COLS_NAMES = ['high', 'low', 'open', 'close', 'vol']
    USE_NAMES = ['high', 'low', 'open', 'close', 'vol', 'average']
    def __init__(self, sname, conf_obj):
        self.sname = sname
        self.filename = conf_obj.get(sname, 'file')
        self.cols = []
        self.use = conf_obj.get(sname, 'use')
        cols = conf_obj.get(sname, 'cols').split(',')
        for c in cols:
            if c.strip() == '':
                continue
            if not c.strip() in self.COLS_NAMES:
                err = "CSV config: col [%s] not in %s" % (c, self.COLS_NAMES)
                raise ConfigError(err)
            self.cols.append(c.strip())
        if not self.use in self.USE_NAMES:
            err = "CSV config: use [%s] not in %s" % (self.use, self.USE_NAMES)
            raise ConfigError(err)
        if self.use != 'average' and not self.use in self.COLS_NAMES:
            err = "CSV config: use [%s] not in cols names and not 'average'" % self.use
            raise ConfigError(err)

    def save(self, conf_obj):
        conf_obj.add_section(self.sname)
        conf_obj.set(self.sname, 'file', self.filename)
        conf_obj.set(self.sname, 'type', self.TYPE)
        clist = ""
        for col in self.cols:
            clist += col + ","
        clist = clist[:-1]
        conf_obj.set(self.sname, 'cols', clist)
        conf_obj.set(self.sname, 'use', self.use)

class LinearSourceConfig(object):
    help = ['section "source.N", "linear"',
            '  Linear source, y = y0 + x * tan(angle).',
            '  parameters:',
            '   - type: linear',
            '   - y0: initial amplitude',
            '   - angle: line angle',
            '   - phase: initial phase [-pi...+pi],']
    TYPE = 'linear'
    ID = 1
    def __init__(self, sname, conf_obj):
        self.sname = sname
        self.y0 = conf_obj.getfloat(self.sname, 'y0')
        self.angle = conf_obj.getfloat(self.sname, 'angle')

    def save(self, conf_obj):
        conf_obj.add_section(self.sname)
        conf_obj.set(self.sname, 'type', self.TYPE)
        conf_obj.set(self.sname, 'y0', self.y0)
        conf_obj.set(self.sname, 'angle', self.angle)


class SinSourceConfig(object):
    help = ['section "source.sin".',
            '  parameters:',
            '   - period: period in ticks along X',
            '   - amp: amplitude',
            '   - phase: initial phase [-180...+180],']
    TYPE = 'sin'
    ID = 2
    def __init__(self, sname, conf_obj):
        self.sname = sname
        self.period = conf_obj.getfloat(self.sname, 'period')
        self.amp = conf_obj.getfloat(self.sname, 'amp')
        self.phase = conf_obj.getint(self.sname, 'phase')

    def save(self, conf_obj):
        conf_obj.add_section(self.sname)
        conf_obj.set(self.sname, 'type', self.TYPE)
        conf_obj.set(self.sname, 'period', self.period)
        conf_obj.set(self.sname, 'amp', self.amp)
        conf_obj.set(self.sname, 'phase', self.phase)


class RandomSourceConfig(object):
    help = ['section "source.N", "random"',
            '  parameters:',
            '   - type: random',
            '   - distrib: one of uniform,normal;',
            '   - min, max - bounds of distribution',
            '   - use_cached - bool. Do not recalc on each step',
            '   Note: "normal not implemented"']
    TYPE = 'random'
    ID = 3
    def __init__(self, sname, conf_obj):
        self.sname = sname
        self.lim_min = conf_obj.getfloat(self.sname, 'min')
        self.lim_max = conf_obj.getfloat(self.sname, 'max')
        self.use_cached = conf_obj.getboolean(self.sname, 'use_cached')
        self.distr_type = 'uniform'
        if conf_obj.has_option(self.sname, 'distrib'):
            self.distr_type = conf_obj.get(sname, 'distrib')
        if not self.distr_type in ['uniform', 'normal']:
            raise ConfigError("unknown distr type [{}]".format(self.distr_type))


    def save(self, conf_obj):
        conf_obj.add_section(self.sname)
        conf_obj.set(self.sname, 'type', self.TYPE)
        conf_obj.set(self.sname, 'min', self.lim_min)
        conf_obj.set(self.sname, 'max', self.lim_max)
        conf_obj.set(self.sname, 'distrib', self.distr_type)
        conf_obj.set(self.sname, 'use_cached', self.use_cached)


class ShowConfig(object):
    help = ['section "show".',
            '  parameters:',
            '   - title - title of view',
            '   - s_show_lim (optional) - limit of X scale']
    SNAME = 'show'
    S_SHOW_LIMITS = ['10', '20', '30', '40', '50', '100']
    def __init__(self, sname, conf_obj):
        self.title = conf_obj.get(self.SNAME, 'title')
        self.s_show_limit = self.S_SHOW_LIMITS[2]   # default
        if conf_obj.has_option(self.SNAME, 's_show_lim'):
            s = conf_obj.getint(self.SNAME, 's_show_lim')
            for sval in self.S_SHOW_LIMITS:
                if s <= int(sval):
                    self.s_show_limit = sval
                    break

    def save(self, conf_obj):
        conf_obj.add_section(self.SNAME)
        conf_obj.set(self.SNAME, 'title', self.title)
        conf_obj.set(self.SNAME, 's_show_lim', self.s_show_limit)

    def get_idx(self):
        ''' get index of current value '''
        for i in range(len(self.S_SHOW_LIMITS)):
            if int(self.s_show_limit) == int(self.S_SHOW_LIMITS[i]):
                return i
        return len(self.S_SHOW_LIMITS) - 1


class SSAConfig(object):
    help = ['section "ssa".',
            '']
    SNAME = 'ssa'
    def __init__(self, sname, conf_obj):
        self.data_range = conf_obj.getint(self.SNAME, 'data_range')
        self.window_size = conf_obj.getint(self.SNAME, 'window_size')
        self.iterations = conf_obj.getint(self.SNAME, 'iterations')
        self.noise_level = conf_obj.getfloat(self.SNAME, 'noise_level')
        self.grouping = conf_obj.get(self.SNAME, 'grouping')

    def save(self, conf_obj):
        conf_obj.add_section(self.SNAME)
        conf_obj.set(self.SNAME, 'data_range', self.data_range)
        conf_obj.set(self.SNAME, 'iterations', self.iterations)
        conf_obj.set(self.SNAME, 'window_size', self.window_size)
        conf_obj.set(self.SNAME, 'noise_level', self.noise_level)
        conf_obj.set(self.SNAME, 'grouping', self.grouping)


class Config(object):
    COMBINED_IDX = 1000  # index when current source is 'combined'
    def __init__(self):
        self.common = None
        self.sources = []
        self.source_idx = -1
        self.ssa = None
        self.show = None
        self.log = logging.getLogger("Config")
        self.fname = ''
        self.errors = ''
        self.sections = {CommonConfig.SNAME : self.make_common_config,
                'source.1' : self.make_source_config,
                'source.2' : self.make_source_config,
                'source.3' : self.make_source_config,
                'source.4' : self.make_source_config,
                'source.5' : self.make_source_config,
                'source.6' : self.make_source_config,
                'source.7' : self.make_source_config,
                'source.9' : self.make_source_config,
                'source.10' : self.make_source_config,
                ShowConfig.SNAME : self.make_show_config,
                SSAConfig.SNAME : self.make_ssa_config}

    def make_source_config(self, sname, cnf):
        tp = cnf.get(sname, 'type')
        if tp == CSVSourceConfig.TYPE:
            self.sources.append(CSVSourceConfig(sname, cnf))
        elif tp == LinearSourceConfig.TYPE:
            self.sources.append(LinearSourceConfig(sname, cnf))
        elif tp == SinSourceConfig.TYPE:
            self.sources.append(SinSourceConfig(sname, cnf))
        elif tp == RandomSourceConfig.TYPE:
            self.sources.append(RandomSourceConfig(sname, cnf))
        else:
            err = "section [%s]: unknown source type '%s'" % (sname, tp)
            self.log.error(err)
            raise ConfigError(err)

    def make_ssa_config(self, sname, cnf):
        if not self.ssa is None:
            raise ConfigError('not unique section %s' % SSAConfig.SNAME)
        self.ssa = SSAConfig(sname, cnf)

    def make_common_config(self, sname, cnf):
        if not self.common is None:
            raise ConfigError('not unique section %s' % CommonConfig.SNAME)
        self.common = CommonConfig(sname, cnf)

    def make_show_config(self, sname, cnf):
        if not self.show is None:
            raise ConfigError('not unique section %s' % ShowConfig.SNAME)
        self.show = ShowConfig(sname, cnf)

    def check_session(self, s, sname):
        if s is None:
            if self.errors == '':
                self.errors = "Missing section(s): " + sname
            else:
                self.errors += ", " + sname

    def parse(self, fname):
        self.common = None
        self.sources = []
        self.ssa = None
        self.show = None
        self.log.info("parsing file [%s]" % fname)
        cnf = configparser.ConfigParser(allow_no_value=True)
#        cnf = ConfigParser.SafeConfigParser()
        cnf.read([fname])
        print("fname: ", fname)
        print("sections: ", cnf.sections())
        for s in cnf.sections():
            print("section ", s)
            if not s in list(self.sections.keys()):
                err = 'Unknown section [%s]' % s
                self.log.error(err)
                raise ConfigError(err)
            # parse section
            self.log.debug("config file - section [%s]" % s)
            self.sections[s](s, cnf)
        self.source_idx = -1
        for i in range(len(self.sources)):
            self.log.debug("Config.parse(): i:%d, self.common.source:"
                    "%s, checked:%s" % (i, self.common.source,
                        self.sources[i].sname))
            if self.sources[i].sname == self.common.source:
                self.source_idx = i
                break
        if self.common.source == 'combined':
            self.source_idx = Config.COMBINED_IDX
        elif self.source_idx < 0:
            err = "Could not find current src [%s]" % self.common.source
            self.log.error(err)
            raise ConfigError(err)
        self.log.debug("Config.parse(): self.source_idx:%d" % self.source_idx)
        self.errors = ''
        self.check_session(self.common, CommonConfig.SNAME)
        self.check_session(self.ssa, SSAConfig.SNAME)
        self.check_session(self.show, ShowConfig.SNAME)
        if self.errors != '':
            self.log.error(self.errors)
            raise ConfigError(self.errors)

    def save(self, fname):
        if self.common is None:
            err = "Could not save - have no data"
            self.log.error(err)
            raise ConfigError(err)
        cnf = configparser.ConfigParser()
        self.common.save(cnf)
        for src in self.sources:
            src.save(cnf)
        self.ssa.save(cnf)
        self.show.save(cnf)
        with open(fname, 'wb') as cfile:
            cnf.write(cfile)
        self.log.info("config saved to file [%s]" % fname)

    def help(self):
        ret = '\nConfiguration parameters\n'
        ret += 'config file divided to sections.\n'
        ret += 'sources numerated as "source.N"\n'
        ret += "Individual source or summ of them can be selected by [source.common], source.\n"
        for cls in [ CommonConfig, CSVSourceConfig, LinearSourceConfig,
                SinSourceConfig, RandomSourceConfig, ShowConfig, SSAConfig]:
            for ln in cls.help:
                ret += ln + '\n'
        return ret



if __name__ == '__main__':

    logging.basicConfig(
            level=logging.DEBUG,
            format="* [%(levelname)8s] [%(name)10s] %(message)s",
            datefmt='%H:%M:%S' )
    log = logging.getLogger('main')
    log.info("Config help:")
    p = Config()
    log.info(p.help())


    f = open('conf_test', 'wb')

    f.write('[source.common]\n')
    f.write('data_size : 18\n')
    f.write('source : source.1\n')
    f.write('predict: 8\n')
    f.write('data_offset: 8\n')

    f.write('[source.1]\n')
    f.write('type: csv\n')
    f.write('file: a.csv\n')
    f.write('cols: high, low\n')
    f.write('use: average\n')

    f.write('[source.3]\n')
    f.write('type: sin\n')
    f.write('period: 22\n')
    f.write('amp: 1.234\n')
    f.write('phase: 45\n')

    f.write('[ssa]\n')
    f.write('data_range: 32\n')
    f.write('window_size : 15\n')
    f.write('iterations: 9\n')
    f.write('noise_level: 0.12\n')
    f.write('grouping:\n')

    f.write('[show]\n')
    f.write('title: test title\n')
    f.write('\n')
    f.close()

    p.parse('conf_test')
    p.save('conf_test2')
    p.parse('conf_test2')


