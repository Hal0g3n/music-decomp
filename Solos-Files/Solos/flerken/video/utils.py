import os
import re
import subprocess
import multiprocessing as mp
import warnings
from collections import OrderedDict

from . import allowed_formats

__all__ = ['get_duration_fps']

#Ripped directly from torchtree, this is awful
class Tree(object):
    def __init__(self):
        self._parameters = 'abrete sesamo'
        self._state_dict_hooks = 'abrete sesamo'
        self._load_state_dict_pre_hooks = 'abrete sesamo'
        self._modules = 'abrete sesamo'
        self._tree_properties = 'abrete sesamo'

        self.set_level(0)

    def level(self):
        return self._tree_properties.get('level')

    def set_level(self, value):
        self._tree_properties.update({'level': value})

    def register_parameter(self, name, param):
        r"""Adds a parameter to the module.
        The parameter can be accessed as an attribute using given name.
        Args:
            name (string): name of the parameter. The parameter can be accessed
                from this module using the given name
            param (Parameter): parameter to be added to the module.
        """
        if '_parameters' not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call")

        elif not isinstance(name, str):
            raise TypeError("parameter name should be a string. "
                            "Got {}".format(type(name)))
        elif '.' in name:
            raise KeyError("parameter name can't contain \".\"")
        elif name == '':
            raise KeyError("parameter name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError("attribute '{}' already exists".format(name))

        if isinstance(param, Tree):
            raise TypeError("parameter cannot be a Tree object. Use add_module to add nodes.")
        else:
            self._parameters[name] = param

    def add_module(self, name, module):
        r"""Adds a child module to the current module.
        The module can be accessed as an attribute using the given name.
        Args:
            name (string): name of the child module. The child module can be
                accessed from this module using the given name
            module (Module): child module to be added to the module.
        """

        if not isinstance(module, Tree) and module is not None:
            raise TypeError("{} is not a Module subclass".format(
                type(module)))
        elif not isinstance(name, str):
            raise TypeError("module name should be a string. Got {}".format(
                type(name)))
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError("attribute '{}' already exists".format(name))
        elif '.' in name:
            raise KeyError("module name can't contain \".\"")
        elif name == '':
            raise KeyError("module name can't be empty string \"\"")
        module.set_level(self.level() + 1)
        self._modules[name] = module

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        for key, buf in self._parameters.items():
            if buf is not None:
                self._parameters[key] = fn(buf)

        return self

    def apply(self, fn):
        r"""Applies ``fn`` recursively to every submodule (as returned by ``.children()``)
        as well as self. Typical use includes initializing the parameters of a model
        (see also :ref:`torch-nn-init`).
        Args:
            fn (:class:`Module` -> None): function to be applied to each submodule
        Returns:
            Module: self
        """
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    def _tracing_name(self, tracing_state):
        if not tracing_state._traced_module_stack:
            return None
        module = tracing_state._traced_module_stack[-1]
        for name, child in module.named_children():
            if child is self:
                return name
        return None

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Support loading old checkpoints that don't have the following attrs:
        if '_state_dict_hooks' not in self.__dict__:
            self._state_dict_hooks = OrderedDict()
        if '_load_state_dict_pre_hooks' not in self.__dict__:
            self._load_state_dict_pre_hooks = OrderedDict()

    def __getattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __setattr__(self, name, value):
        def remove_from(*dicts):
            for d in dicts:
                if name in d:
                    del d[name]

        if value == 'abrete sesamo':
            object.__setattr__(self, name, OrderedDict())


        elif isinstance(value, Tree):
            modules = self.__dict__.get('_modules')
            if modules is None:
                raise AttributeError(
                    "cannot assign module before Module.__init__() call")
            remove_from(self.__dict__, self._parameters)
            modules[name] = value
        else:
            params = self.__dict__.get('_parameters')
            if params is None:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__() call")
            remove_from(self.__dict__, self._modules)
            self.register_parameter(name, value)

    def __delattr__(self, name):
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._modules:
            del self._modules[name]
        else:
            object.__delattr__(self, name)
    def __call__(self, *args):
        tmp = self
        for param in args:
            tmp = tmp.__getattr__(param)
        return tmp
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        r"""Returns a dictionary containing a whole state of the module.
        Both parameters and persistent buffers (e.g. running averages) are
        included. Keys are corresponding parameter and buffer names.
        Returns:
            dict:
                a dictionary containing a whole state of the module
        Example::
            >>> module.state_dict().keys()
            ['bias', 'weight']
        """
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        destination._metadata[prefix[:-1]] = local_metadata = dict(level=self.level())
        for name, param in self._parameters.items():
            if param is not None:
                destination[prefix + name] = param
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination, prefix + name + '.', keep_vars=keep_vars)
        for hook in self._state_dict_hooks.values():
            hook_result = hook(self, destination, prefix, local_metadata)
            if hook_result is not None:
                destination = hook_result
        return destination

    def parameters(self, recurse=True):
        r"""Returns an iterator over module parameters.
        This is typically passed to an optimizer.
        Args:
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.
        Yields:
            Parameter: module parameter
        """
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(self, prefix='', recurse=True):
        r"""Returns an iterator over module parameters, yielding both the
        name of the parameter as well as the parameter itself.
        Args:
            prefix (str): prefix to prepend to all parameter names.
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.
        Yields:
            (string, Parameter): Tuple containing the name and parameter
        """
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def named_children(self):
        r"""Returns an iterator over immediate children modules, yielding both
        the name of the module as well as the module itself.
        Yields:
            (string, Module): Tuple containing a name and child module
        """
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def modules(self):
        r"""Returns an iterator over all modules in the network.
        Yields:
            Module: a module in the network
        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.
        """
        for name, module in self.named_modules():
            yield module

    def children(self):
        r"""Returns an iterator over immediate children modules.
        Yields:
            Module: a child module
        """
        for name, module in self.named_children():
            yield module

    def _named_members(self, get_members_fn, prefix='', recurse=True):
        r"""Helper method for yielding various names + members of modules."""
        memo = set()
        modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                yield name, v

    def named_modules(self, memo=None, prefix=''):
        r"""Returns an iterator over all modules in the network, yielding
        both the name of the module as well as the module itself.
        Yields:
            (string, Module): Tuple of name and module
        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.
        """

        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                for m in module.named_modules(memo, submodule_prefix):
                    yield m

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str

    def extra_repr(self):
        r"""Set the extra representation of the module
        To print customized extra information, you should reimplement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
        return ''

    def _get_name(self):
        return self.__class__.__name__

    def __dir__(self):
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        parameters = list(self._parameters.keys())
        modules = list(self._modules.keys())
        keys = module_attrs + attrs + parameters + modules

        # Eliminate attrs that are not legal Python variable names
        keys = [key for key in keys if not key[0].isdigit()]

        return sorted(keys)

import keyword

RESERVED = keyword.kwlist
RESERVED.extend(list(object.__dict__.keys()))
RESERVED.extend(['__pycache__'])

from os import scandir
import os

def scantree(path, tree):
    """Recursively yield DirEntry objects for given directory."""
    for entry in scandir(path):
        if entry.name[0] != '.' and os.path.splitext(entry.name)[0] not in RESERVED:
            if entry.is_dir(follow_symlinks=False):
                tree.add_module(entry.name, Directory_Tree())
                yield from scantree(entry.path, getattr(tree, entry.name))
            else:
                tree.register_parameter(os.path.splitext(entry.name)[0], os.path.splitext(entry.name)[1])
                yield entry

class Directory_Tree(Tree):
    def __init__(self, path=None):
        super(Directory_Tree, self).__init__()
        if path is not None:
            list(scantree(path, self))

    def _named_members(self, get_members_fn, prefix='', recurse=True):
        r"""Helper method for yielding various names + members of modules."""
        memo = set()
        modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                memo.add(v)
                name = module_prefix + ('/' if module_prefix else '') + k
                yield name, v

    def named_modules(self, memo=None, prefix=''):
        r"""Returns an iterator over all modules in the network, yielding
        both the name of the module as well as the module itself.
        Yields:
            (string, Module): Tuple of name and module
        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.
        """

        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ('/' if prefix else '') + name
                for m in module.named_modules(memo, submodule_prefix):
                    yield m

    def clone_tree(self, path):
        r"""
        Clones the tree directory into given path.
        :param path: Relative root in which tree directory will be cloned
        :return: None
        """
        for module, _ in self.named_modules():
            _path = os.path.join(path, module)
            if module != '' and not os.path.exists(_path):
                os.mkdir(_path)

    def paths(self, root='', recurse=True):
        r"""Returns an iterator over module parameters, yielding both the
        name of the parameter as well as the parameter itself.
        Args:
            root (str): prefix to prepend to all parameter names.
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.
        Yields:
            (string, Parameter): Tuple containing the name and parameter
        """
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=root, recurse=recurse)
        for elem in gen:
            yield elem[0] + elem[1]


def get_duration_fps(filename, display):
    """
    Wraps ffprobe to get file duration and fps

    :param filename: str, Path to file to be evaluate
    :param display: ['ms','s','min','h'] Time format miliseconds, sec, minutes, hours.
    :return: tuple(time, fps) in the mentioned format
    """

    def ffprobe2ms(time):
        cs = int(time[-2::])
        s = int(os.path.splitext(time[-5::])[0])
        idx = time.find(':')
        h = int(time[0:idx - 1])
        m = int(time[idx + 1:idx + 3])
        return [h, m, s, cs]

    # Get length of video with filename
    time = None
    fps = None
    result = subprocess.Popen(["ffprobe", str(filename)],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = [str(x) for x in result.stdout.readlines()]
    info_lines = [x for x in output if "Duration:" in x or "Stream" in x]
    duration_line = [x for x in info_lines if "Duration:" in x]
    fps_line = [x for x in info_lines if "Stream" in x]
    if duration_line:
        duration_str = duration_line[0].split(",")[0]
        pattern = '\d{2}:\d{2}:\d{2}.\d{2}'
        dt = re.findall(pattern, duration_str)[0]
        time = ffprobe2ms(dt)
    if fps_line:
        pattern = '(\d{2})(.\d{2})* fps'
        fps_elem = re.findall(pattern, fps_line[0])[0]
        fps = float(fps_elem[0] + fps_elem[1])
    if display == 's':
        time = time[0] * 3600 + time[1] * 60 + time[2] + time[3] / 100.0
    elif display == 'ms':
        time = (time[0] * 3600 + time[1] * 60 + time[2] + time[3] / 100.0) * 1000
    elif display == 'min':
        time = (time[0] * 3600 + time[1] * 60 + time[2] + time[3] / 100.0) / 60
    elif display == 'h':
        time = (time[0] * 3600 + time[1] * 60 + time[2] + time[3] / 100.0) / 3600
    return (time, fps)


def reencode_25_interpolate(video_path: str, dst_path: str, *args, **kwargs):
    kwargs['input_options']=['-y']
    kwargs['output_options']=['-r', '25']
    return apply_single(video_path, dst_path, *args, **kwargs)


def reencode_30_interpolate(video_path: str, dst_path: str, *args, **kwargs):
    input_options = ['-y']
    output_options = ['-r', '30']
    return apply_single(video_path, dst_path, input_options, output_options, *args, **kwargs)


def apply_single(video_path: str, dst_path: str, input_options: list, output_options: list, ext: None):
    """
    Runs ffmpeg for the following format for a single input/output:
        ffmpeg [input options] -i input [output options] output


    :param video_path: str Path to input video
    :param dst_path: str Path to output video
    :param input_options: List[str] list of ffmpeg options ready for a Popen format
    :param output_options: List[str] list of ffmpeg options ready for a Popen format
    :return: None
    """
    assert os.path.isfile(video_path)
    assert os.path.isdir(os.path.dirname(dst_path))
    if ext is not None:
        dst_path = os.path.splitext(dst_path)[0] + ext
    result = subprocess.Popen(["ffmpeg", *input_options, '-i', video_path, *output_options, dst_path],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout = result.stdout.read()
    stderr = result.stdout.read()
    if stdout is not None:
        print(stdout.decode("utf-8"))
    if stderr is not None:
        print(stderr.decode("utf-8"))


def apply_tree(root, dst, input_options=list(), output_options=list(), multiprocessing=0, fn=apply_single, ignore=[],
               ext=None):
    """
    Applies ffmpeg processing for a given directory tree for whatever which fits this format:
        ffmpeg [input options] -i input [output options] output
    System automatically checks if files in directory are ffmpeg compatible
    Results will be stored in a replicated tree with same structure and filenames


    :param root: Root directory in which files are stored
    :param dst: Destiny directory in which to store results
    :param input_options: list[str] ffmpeg input options in a subprocess format
    :param output_options: list[str] ffmpeg output options in a subprocess format
    :param multiprocessing: int if 0 disables multiprocessin, else enables multiprocessing with that amount of cores.
    :param fn: funcion to be used. By default requires I/O options.
    :return: None
    """
    formats = allowed_formats()  # List of ffmpeg compatible formats
    tree = Directory_Tree(root)  # Directory tree
    if not os.path.exists(dst):
        os.mkdir(dst)
    tree.clone_tree(dst)  # Generates new directory structure (folders)

    # Python Multiproceesing mode
    if multiprocessing > 0:
        pool = mp.Pool(multiprocessing)
        results = [pool.apply(fn,
                              args=(i_path, o_path),
                              kwds={input_options: input_options, output_options: output_options, ext: ext})
                   for i_path, o_path in zip(tree.paths(root), tree.paths(dst)) if
                   os.path.splitext(i_path)[1][1:] in formats]
        pool.close()
    else:
        for i_path, o_path in zip(tree.paths(root), tree.paths(dst)):
            if os.path.splitext(i_path)[1][1:] in formats:
                fn(i_path, o_path, input_options=input_options, output_options=output_options, ext=ext)


def quirurgical_extractor(video_path, dst_frames, dst_audio, t, n_frames, T, size=None, sample_rate=None):
    """
    FFMPEG wraper which extracts, from an instant t onwards, n_frames (jpg) and T seconds of audio (wav).
    Optionally performs frame resizing and audio resampling.


    :param video_path: str path to input video
    :param dst_frames: str path to folder in which to store outgoing frames (If doesn't exist will be created)
    :param dst_audio: str path to folder in which to store outgoing audio (If doesn't exist will be created)
    :param t: Initial time from which start the extraction
    :param n_frames: Amount of frames to be extracted from t
    :param T: amount of seconds of audio to be extracted from t
    :param size: str (optional) frame resizing (ej: '256x256')
    :param sample_rate: (optional) audio resampling
    :return: List[str] line-wise console output
    """

    #    dst_frames = os.path.join(dst_frames, '{0:05d}'.format(t))
    if not os.path.exists(dst_audio):
        os.makedirs(dst_audio)
    dst_audio = os.path.join(dst_audio, '{0:05d}.wav'.format(t))
    if not os.path.exists(dst_frames):
        os.makedirs(dst_frames)
    stream = ['ffmpeg', '-y', '-ss', str(t), '-i', video_path]
    if size is not None:
        stream.extend(['-s', size])
    stream.extend(['-frames:v', str(n_frames), dst_frames + '/%02d.png', '-vn', '-ac', '1'])
    if sample_rate is not None:
        stream.extend(['-ar', str(sample_rate)])
    stream.extend(['-t', str(T), dst_audio])

    result = subprocess.Popen(stream,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout = result.stdout.read()
    stderr = result.stdout.read()
    if stdout is not None:
        print(stdout.decode("utf-8"))
    if stderr is not None:
        print(stderr.decode("utf-8"))


def quirurgical_extractor_tree(root, dst, n_frames, T, size=None, sample_rate=None, multiprocessing=0,
                               stamp_generator_fn=None, formats=None, ignore=[]):
    """
    Quirurgical extractor over a directory tree. Clones directory tree in dst folder twice, once for audio and another
    time for frames. For each recording there will be a folder with the same name in dst directory containing audio segm
    ments in case of audio or folders with frames in case of video.


    :param root: Original path of recordings
    :param dst: Destiny path
    :param n_frames: Amount of frames to get for each instant
    :param T: Amount of time (seconds) of audio to get for each instant
    :param size: Optinal image resizing for all frames
    :param sample_rate: Optinal audio resampling for all tracks
    :param multiprocessing: Whether to use or not multiprocessing, By default, 0 , disables MP.
     Else selects amount of cores
    :param stamp_generator_fn: By default function generate time stamps in wich to cut. It is possible to pass
    custom function that takes as input recording path and returns time in seconds (integer). As suggestion you may
    use a dictionary by passing dict.get method as generator function.
    :param formats: List of strings indicating allowed formats. Only files with chosen formats will be processed. By
    default this list is taken from ffmpeg -formats.
    :param ignore: list of folders to be ignored at the time ob building directory tree
    """

    def stamp_generator(video_path):
        time, fps = get_duration_fps(video_path, 's')
        segments = range(0, int(time) - T, T)
        if T * fps != n_frames:
            warnings.warn('Warning: n_frames != fps * T')
        return segments

    def path_stamp_generator(path, gen, T):
        for x in gen:
            yield (os.path.join(path, '{0:05d}to{1:05d}'.format(x, x + T)), x)

    if stamp_generator_fn is None:
        stamp_generator_fn = stamp_generator
    if formats is None:
        formats = allowed_formats()  # List of ffmpeg compatible formats
    tree = Directory_Tree(root, ignore=ignore)  # Directory tree
    """
    Tree generates an object-based tree of the directory treem, where folders are nodes and parameters are files.
    The idea is to replicate this directory tree for audio files and video files in audio_root and video_root 
    respectively.
    """
    audio_root = os.path.join(dst, 'audio')
    video_root = os.path.join(dst, 'frames')
    if not os.path.exists(audio_root):
        os.makedirs(audio_root)  # Create directory
    if not os.path.exists(video_root):
        os.makedirs(video_root)
    tree.clone_tree(audio_root)  # Clone directory tree structure in for each one
    tree.clone_tree(video_root)

    """
    There is an important concept to get:
    For each video, we will obtain S segments, nevertheless, each segment contains different amount of files depending
    on if it's audio or frames.

    For frames we get S*n_frames frames, which are distributed in S folders. This means that for each video there will 
    be S folders containing n_frames frames.

    On contrary, for audio we will obtain S files, therefore there will be  a single folder per video containing S files

    Be aware video folders have to be created dynamically depending on S, meanwhile audio directory tree can be
    precomputed based on amount of recording.

    """

    """
    These generator generate relative paths to all files existing in the directory. Therefore we obtain a paths to all
    recordings, paths to audio folders (1 folder per recording) and path to video folders. We still have to generate S
    video subfolders for all the segments
    """
    i_path_generator = tree.named_parameters(root)  # Generator of input (path,format)
    a_path_generator = tree.named_parameters(audio_root)  # Generator of audio-wise output paths
    v_path_generator = tree.named_parameters(video_root)  # Generator of video-wise output paths

    failure_list = []
    for i_path, a_path, v_path in zip(i_path_generator, a_path_generator, v_path_generator):
        # This for loop provides needed paths recording-wise
        try:
            if i_path[1][1:] in formats:  # Check file has a valid ffmpeg format.
                stamp_gen = stamp_generator_fn(i_path[0] + i_path[1])
                path_stamp_gen = path_stamp_generator(v_path[0], stamp_gen, T)
                """
                Generator that calculate amount of segments,t,  and dynamically generates S video subfolder paths.
                Obviously, this generator provides t segments and t subfolder names.
                """
                if multiprocessing == 0:
                    for path, t in path_stamp_gen:
                        quirurgical_extractor(i_path[0] + i_path[1], path, a_path[0], t, n_frames, T, size, sample_rate)
                else:
                    pool = mp.Pool(multiprocessing)
                    results = [pool.apply(quirurgical_extractor,
                                          args=(
                                              i_path[0] + i_path[1], path, a_path[0], t, n_frames, T, size,
                                              sample_rate))
                               for path, t in path_stamp_gen]
                    pool.close()
        except:
            failure_list.append(i_path)
    print('Following videos have failed (from python side):\n')
    for v in failure_list:
        print('{0} \n'.format(v))
    return failure_list
