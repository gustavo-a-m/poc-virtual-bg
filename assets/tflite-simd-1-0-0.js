var createTFLiteSIMDModule = (function () {
  var _scriptDir =
    typeof document !== 'undefined' && document.currentScript
      ? document.currentScript.src
      : undefined;
  if (typeof __filename !== 'undefined') _scriptDir = _scriptDir || __filename;
  return function (createTwilioTFLiteSIMDModule) {
    createTwilioTFLiteSIMDModule = createTwilioTFLiteSIMDModule || {};

    var Module =
      typeof createTwilioTFLiteSIMDModule !== 'undefined'
        ? createTwilioTFLiteSIMDModule
        : {};
    var readyPromiseResolve, readyPromiseReject;
    Module['ready'] = new Promise(function (resolve, reject) {
      readyPromiseResolve = resolve;
      readyPromiseReject = reject;
    });
    var moduleOverrides = {};
    var key;
    for (key in Module) {
      if (Module.hasOwnProperty(key)) {
        moduleOverrides[key] = Module[key];
      }
    }
    var arguments_ = [];
    var thisProgram = './this.program';
    var quit_ = function (status, toThrow) {
      throw toThrow;
    };
    var ENVIRONMENT_IS_WEB = false;
    var ENVIRONMENT_IS_WORKER = false;
    var ENVIRONMENT_IS_NODE = false;
    var ENVIRONMENT_IS_SHELL = false;
    ENVIRONMENT_IS_WEB = typeof window === 'object';
    ENVIRONMENT_IS_WORKER = typeof importScripts === 'function';
    ENVIRONMENT_IS_NODE =
      typeof process === 'object' &&
      typeof process.versions === 'object' &&
      typeof process.versions.node === 'string';
    ENVIRONMENT_IS_SHELL =
      !ENVIRONMENT_IS_WEB && !ENVIRONMENT_IS_NODE && !ENVIRONMENT_IS_WORKER;
    var scriptDirectory = '';
    function locateFile(path) {
      if (Module['locateFile']) {
        return Module['locateFile'](path, scriptDirectory);
      }
      return scriptDirectory + path;
    }
    var read_, readAsync, readBinary, setWindowTitle;
    var nodeFS;
    var nodePath;
    if (ENVIRONMENT_IS_NODE) {
      if (ENVIRONMENT_IS_WORKER) {
        scriptDirectory = require('path').dirname(scriptDirectory) + '/';
      } else {
        scriptDirectory = __dirname + '/';
      }
      read_ = function shell_read(filename, binary) {
        if (!nodeFS) nodeFS = require('fs');
        if (!nodePath) nodePath = require('path');
        filename = nodePath['normalize'](filename);
        return nodeFS['readFileSync'](filename, binary ? null : 'utf8');
      };
      readBinary = function readBinary(filename) {
        var ret = read_(filename, true);
        if (!ret.buffer) {
          ret = new Uint8Array(ret);
        }
        assert(ret.buffer);
        return ret;
      };
      if (process['argv'].length > 1) {
        thisProgram = process['argv'][1].replace(/\\/g, '/');
      }
      arguments_ = process['argv'].slice(2);
      process['on']('uncaughtException', function (ex) {
        if (!(ex instanceof ExitStatus)) {
          throw ex;
        }
      });
      process['on']('unhandledRejection', abort);
      quit_ = function (status) {
        process['exit'](status);
      };
      Module['inspect'] = function () {
        return '[Emscripten Module object]';
      };
    } else if (ENVIRONMENT_IS_SHELL) {
      if (typeof read != 'undefined') {
        read_ = function shell_read(f) {
          return read(f);
        };
      }
      readBinary = function readBinary(f) {
        var data;
        if (typeof readbuffer === 'function') {
          return new Uint8Array(readbuffer(f));
        }
        data = read(f, 'binary');
        assert(typeof data === 'object');
        return data;
      };
      if (typeof scriptArgs != 'undefined') {
        arguments_ = scriptArgs;
      } else if (typeof arguments != 'undefined') {
        arguments_ = arguments;
      }
      if (typeof quit === 'function') {
        quit_ = function (status) {
          quit(status);
        };
      }
      if (typeof print !== 'undefined') {
        if (typeof console === 'undefined') console = {};
        console.log = print;
        console.warn = console.error =
          typeof printErr !== 'undefined' ? printErr : print;
      }
    } else if (ENVIRONMENT_IS_WEB || ENVIRONMENT_IS_WORKER) {
      if (ENVIRONMENT_IS_WORKER) {
        scriptDirectory = self.location.href;
      } else if (typeof document !== 'undefined' && document.currentScript) {
        scriptDirectory = document.currentScript.src;
      }
      if (_scriptDir) {
        scriptDirectory = _scriptDir;
      }
      if (scriptDirectory.indexOf('blob:') !== 0) {
        scriptDirectory = scriptDirectory.substr(
          0,
          scriptDirectory.lastIndexOf('/') + 1
        );
      } else {
        scriptDirectory = '';
      }
      {
        read_ = function (url) {
          var xhr = new XMLHttpRequest();
          xhr.open('GET', url, false);
          xhr.send(null);
          return xhr.responseText;
        };
        if (ENVIRONMENT_IS_WORKER) {
          readBinary = function (url) {
            var xhr = new XMLHttpRequest();
            xhr.open('GET', url, false);
            xhr.responseType = 'arraybuffer';
            xhr.send(null);
            return new Uint8Array(xhr.response);
          };
        }
        readAsync = function (url, onload, onerror) {
          var xhr = new XMLHttpRequest();
          xhr.open('GET', url, true);
          xhr.responseType = 'arraybuffer';
          xhr.onload = function () {
            if (xhr.status == 200 || (xhr.status == 0 && xhr.response)) {
              onload(xhr.response);
              return;
            }
            onerror();
          };
          xhr.onerror = onerror;
          xhr.send(null);
        };
      }
      setWindowTitle = function (title) {
        document.title = title;
      };
    } else {
    }
    var out = Module['print'] || console.log.bind(console);
    var err = Module['printErr'] || console.warn.bind(console);
    for (key in moduleOverrides) {
      if (moduleOverrides.hasOwnProperty(key)) {
        Module[key] = moduleOverrides[key];
      }
    }
    moduleOverrides = null;
    if (Module['arguments']) arguments_ = Module['arguments'];
    if (Module['thisProgram']) thisProgram = Module['thisProgram'];
    if (Module['quit']) quit_ = Module['quit'];
    var wasmBinary;
    if (Module['wasmBinary']) wasmBinary = Module['wasmBinary'];
    var noExitRuntime = Module['noExitRuntime'] || true;
    if (typeof WebAssembly !== 'object') {
      abort('no native wasm support detected');
    }
    var wasmMemory;
    var ABORT = false;
    var EXITSTATUS;
    function assert(condition, text) {
      if (!condition) {
        abort('Assertion failed: ' + text);
      }
    }
    var UTF8Decoder =
      typeof TextDecoder !== 'undefined' ? new TextDecoder('utf8') : undefined;
    function UTF8ArrayToString(heap, idx, maxBytesToRead) {
      var endIdx = idx + maxBytesToRead;
      var endPtr = idx;
      while (heap[endPtr] && !(endPtr >= endIdx)) ++endPtr;
      if (endPtr - idx > 16 && heap.subarray && UTF8Decoder) {
        return UTF8Decoder.decode(heap.subarray(idx, endPtr));
      } else {
        var str = '';
        while (idx < endPtr) {
          var u0 = heap[idx++];
          if (!(u0 & 128)) {
            str += String.fromCharCode(u0);
            continue;
          }
          var u1 = heap[idx++] & 63;
          if ((u0 & 224) == 192) {
            str += String.fromCharCode(((u0 & 31) << 6) | u1);
            continue;
          }
          var u2 = heap[idx++] & 63;
          if ((u0 & 240) == 224) {
            u0 = ((u0 & 15) << 12) | (u1 << 6) | u2;
          } else {
            u0 = ((u0 & 7) << 18) | (u1 << 12) | (u2 << 6) | (heap[idx++] & 63);
          }
          if (u0 < 65536) {
            str += String.fromCharCode(u0);
          } else {
            var ch = u0 - 65536;
            str += String.fromCharCode(55296 | (ch >> 10), 56320 | (ch & 1023));
          }
        }
      }
      return str;
    }
    function UTF8ToString(ptr, maxBytesToRead) {
      return ptr ? UTF8ArrayToString(HEAPU8, ptr, maxBytesToRead) : '';
    }
    function writeAsciiToMemory(str, buffer, dontAddNull) {
      for (var i = 0; i < str.length; ++i) {
        HEAP8[buffer++ >> 0] = str.charCodeAt(i);
      }
      if (!dontAddNull) HEAP8[buffer >> 0] = 0;
    }
    function alignUp(x, multiple) {
      if (x % multiple > 0) {
        x += multiple - (x % multiple);
      }
      return x;
    }
    var buffer,
      HEAP8,
      HEAPU8,
      HEAP16,
      HEAPU16,
      HEAP32,
      HEAPU32,
      HEAPF32,
      HEAPF64;
    function updateGlobalBufferAndViews(buf) {
      buffer = buf;
      Module['HEAP8'] = HEAP8 = new Int8Array(buf);
      Module['HEAP16'] = HEAP16 = new Int16Array(buf);
      Module['HEAP32'] = HEAP32 = new Int32Array(buf);
      Module['HEAPU8'] = HEAPU8 = new Uint8Array(buf);
      Module['HEAPU16'] = HEAPU16 = new Uint16Array(buf);
      Module['HEAPU32'] = HEAPU32 = new Uint32Array(buf);
      Module['HEAPF32'] = HEAPF32 = new Float32Array(buf);
      Module['HEAPF64'] = HEAPF64 = new Float64Array(buf);
    }
    var INITIAL_MEMORY = Module['INITIAL_MEMORY'] || 16777216;
    var wasmTable;
    var __ATPRERUN__ = [];
    var __ATINIT__ = [];
    var __ATMAIN__ = [];
    var __ATPOSTRUN__ = [];
    var runtimeInitialized = false;
    var runtimeExited = false;
    function preRun() {
      if (Module['preRun']) {
        if (typeof Module['preRun'] == 'function')
          Module['preRun'] = [Module['preRun']];
        while (Module['preRun'].length) {
          addOnPreRun(Module['preRun'].shift());
        }
      }
      callRuntimeCallbacks(__ATPRERUN__);
    }
    function initRuntime() {
      runtimeInitialized = true;
      callRuntimeCallbacks(__ATINIT__);
    }
    function preMain() {
      callRuntimeCallbacks(__ATMAIN__);
    }
    function exitRuntime() {
      runtimeExited = true;
    }
    function postRun() {
      if (Module['postRun']) {
        if (typeof Module['postRun'] == 'function')
          Module['postRun'] = [Module['postRun']];
        while (Module['postRun'].length) {
          addOnPostRun(Module['postRun'].shift());
        }
      }
      callRuntimeCallbacks(__ATPOSTRUN__);
    }
    function addOnPreRun(cb) {
      __ATPRERUN__.unshift(cb);
    }
    function addOnInit(cb) {
      __ATINIT__.unshift(cb);
    }
    function addOnPostRun(cb) {
      __ATPOSTRUN__.unshift(cb);
    }
    var runDependencies = 0;
    var runDependencyWatcher = null;
    var dependenciesFulfilled = null;
    function addRunDependency(id) {
      runDependencies++;
      if (Module['monitorRunDependencies']) {
        Module['monitorRunDependencies'](runDependencies);
      }
    }
    function removeRunDependency(id) {
      runDependencies--;
      if (Module['monitorRunDependencies']) {
        Module['monitorRunDependencies'](runDependencies);
      }
      if (runDependencies == 0) {
        if (runDependencyWatcher !== null) {
          clearInterval(runDependencyWatcher);
          runDependencyWatcher = null;
        }
        if (dependenciesFulfilled) {
          var callback = dependenciesFulfilled;
          dependenciesFulfilled = null;
          callback();
        }
      }
    }
    Module['preloadedImages'] = {};
    Module['preloadedAudios'] = {};
    function abort(what) {
      if (Module['onAbort']) {
        Module['onAbort'](what);
      }
      what += '';
      err(what);
      ABORT = true;
      EXITSTATUS = 1;
      what = 'abort(' + what + '). Build with -s ASSERTIONS=1 for more info.';
      var e = new WebAssembly.RuntimeError(what);
      readyPromiseReject(e);
      throw e;
    }
    function hasPrefix(str, prefix) {
      return String.prototype.startsWith
        ? str.startsWith(prefix)
        : str.indexOf(prefix) === 0;
    }
    var dataURIPrefix = 'data:application/octet-stream;base64,';
    function isDataURI(filename) {
      return hasPrefix(filename, dataURIPrefix);
    }
    var fileURIPrefix = 'file://';
    function isFileURI(filename) {
      return hasPrefix(filename, fileURIPrefix);
    }
    var wasmBinaryFile = 'tflite-simd-1-0-0.wasm';
    if (!isDataURI(wasmBinaryFile)) {
      wasmBinaryFile = locateFile(wasmBinaryFile);
    }
    function getBinary(file) {
      try {
        if (file == wasmBinaryFile && wasmBinary) {
          return new Uint8Array(wasmBinary);
        }
        if (readBinary) {
          return readBinary(file);
        } else {
          throw 'both async and sync fetching of the wasm failed';
        }
      } catch (err) {
        abort(err);
      }
    }
    function getBinaryPromise() {
      if (!wasmBinary && (ENVIRONMENT_IS_WEB || ENVIRONMENT_IS_WORKER)) {
        if (typeof fetch === 'function' && !isFileURI(wasmBinaryFile)) {
          return fetch(wasmBinaryFile, { credentials: 'same-origin' })
            .then(function (response) {
              if (!response['ok']) {
                throw (
                  "failed to load wasm binary file at '" + wasmBinaryFile + "'"
                );
              }
              return response['arrayBuffer']();
            })
            .catch(function () {
              return getBinary(wasmBinaryFile);
            });
        } else {
          if (readAsync) {
            return new Promise(function (resolve, reject) {
              readAsync(
                wasmBinaryFile,
                function (response) {
                  resolve(new Uint8Array(response));
                },
                reject
              );
            });
          }
        }
      }
      return Promise.resolve().then(function () {
        return getBinary(wasmBinaryFile);
      });
    }
    function createWasm() {
      var info = { a: asmLibraryArg };
      function receiveInstance(instance, module) {
        var exports = instance.exports;
        Module['asm'] = exports;
        wasmMemory = Module['asm']['q'];
        updateGlobalBufferAndViews(wasmMemory.buffer);
        wasmTable = Module['asm']['D'];
        addOnInit(Module['asm']['r']);
        removeRunDependency('wasm-instantiate');
      }
      addRunDependency('wasm-instantiate');
      function receiveInstantiatedSource(output) {
        receiveInstance(output['instance']);
      }
      function instantiateArrayBuffer(receiver) {
        return getBinaryPromise()
          .then(function (binary) {
            var result = WebAssembly.instantiate(binary, info);
            return result;
          })
          .then(receiver, function (reason) {
            err('failed to asynchronously prepare wasm: ' + reason);
            abort(reason);
          });
      }
      function instantiateAsync() {
        if (
          !wasmBinary &&
          typeof WebAssembly.instantiateStreaming === 'function' &&
          !isDataURI(wasmBinaryFile) &&
          !isFileURI(wasmBinaryFile) &&
          typeof fetch === 'function'
        ) {
          return fetch(wasmBinaryFile, { credentials: 'same-origin' }).then(
            function (response) {
              var result = WebAssembly.instantiateStreaming(response, info);
              return result.then(receiveInstantiatedSource, function (reason) {
                err('wasm streaming compile failed: ' + reason);
                err('falling back to ArrayBuffer instantiation');
                return instantiateArrayBuffer(receiveInstantiatedSource);
              });
            }
          );
        } else {
          return instantiateArrayBuffer(receiveInstantiatedSource);
        }
      }
      if (Module['instantiateWasm']) {
        try {
          var exports = Module['instantiateWasm'](info, receiveInstance);
          return exports;
        } catch (e) {
          err('Module.instantiateWasm callback failed with error: ' + e);
          return false;
        }
      }
      instantiateAsync().catch(readyPromiseReject);
      return {};
    }
    function callRuntimeCallbacks(callbacks) {
      while (callbacks.length > 0) {
        var callback = callbacks.shift();
        if (typeof callback == 'function') {
          callback(Module);
          continue;
        }
        var func = callback.func;
        if (typeof func === 'number') {
          if (callback.arg === undefined) {
            wasmTable.get(func)();
          } else {
            wasmTable.get(func)(callback.arg);
          }
        } else {
          func(callback.arg === undefined ? null : callback.arg);
        }
      }
    }
    var runtimeKeepaliveCounter = 0;
    function keepRuntimeAlive() {
      return noExitRuntime || runtimeKeepaliveCounter > 0;
    }
    function _abort() {
      abort();
    }
    var _emscripten_get_now;
    if (ENVIRONMENT_IS_NODE) {
      _emscripten_get_now = function () {
        var t = process['hrtime']();
        return t[0] * 1e3 + t[1] / 1e6;
      };
    } else if (typeof dateNow !== 'undefined') {
      _emscripten_get_now = dateNow;
    } else
      _emscripten_get_now = function () {
        return performance.now();
      };
    var _emscripten_get_now_is_monotonic = true;
    function setErrNo(value) {
      HEAP32[___errno_location() >> 2] = value;
      return value;
    }
    function _clock_gettime(clk_id, tp) {
      var now;
      if (clk_id === 0) {
        now = Date.now();
      } else if (
        (clk_id === 1 || clk_id === 4) &&
        _emscripten_get_now_is_monotonic
      ) {
        now = _emscripten_get_now();
      } else {
        setErrNo(28);
        return -1;
      }
      HEAP32[tp >> 2] = (now / 1e3) | 0;
      HEAP32[(tp + 4) >> 2] = ((now % 1e3) * 1e3 * 1e3) | 0;
      return 0;
    }
    function _dlopen(filename, flag) {
      abort(
        "To use dlopen, you need to use Emscripten's linking support, see https://github.com/emscripten-core/emscripten/wiki/Linking"
      );
    }
    function _dlsym(handle, symbol) {
      abort(
        "To use dlopen, you need to use Emscripten's linking support, see https://github.com/emscripten-core/emscripten/wiki/Linking"
      );
    }
    function _emscripten_get_heap_max() {
      return 2147483648;
    }
    function _emscripten_memcpy_big(dest, src, num) {
      HEAPU8.copyWithin(dest, src, src + num);
    }
    function emscripten_realloc_buffer(size) {
      try {
        wasmMemory.grow((size - buffer.byteLength + 65535) >>> 16);
        updateGlobalBufferAndViews(wasmMemory.buffer);
        return 1;
      } catch (e) {}
    }
    function _emscripten_resize_heap(requestedSize) {
      var oldSize = HEAPU8.length;
      var maxHeapSize = 2147483648;
      if (requestedSize > maxHeapSize) {
        return false;
      }
      for (var cutDown = 1; cutDown <= 4; cutDown *= 2) {
        var overGrownHeapSize = oldSize * (1 + 0.2 / cutDown);
        overGrownHeapSize = Math.min(
          overGrownHeapSize,
          requestedSize + 100663296
        );
        var newSize = Math.min(
          maxHeapSize,
          alignUp(Math.max(requestedSize, overGrownHeapSize), 65536)
        );
        var replacement = emscripten_realloc_buffer(newSize);
        if (replacement) {
          return true;
        }
      }
      return false;
    }
    function _emscripten_thread_sleep(msecs) {
      var start = _emscripten_get_now();
      while (_emscripten_get_now() - start < msecs) {}
    }
    var ENV = {};
    function getExecutableName() {
      return thisProgram || './this.program';
    }
    function getEnvStrings() {
      if (!getEnvStrings.strings) {
        var lang =
          (
            (typeof navigator === 'object' &&
              navigator.languages &&
              navigator.languages[0]) ||
            'C'
          ).replace('-', '_') + '.UTF-8';
        var env = {
          USER: 'web_user',
          LOGNAME: 'web_user',
          PATH: '/',
          PWD: '/',
          HOME: '/home/web_user',
          LANG: lang,
          _: getExecutableName(),
        };
        for (var x in ENV) {
          env[x] = ENV[x];
        }
        var strings = [];
        for (var x in env) {
          strings.push(x + '=' + env[x]);
        }
        getEnvStrings.strings = strings;
      }
      return getEnvStrings.strings;
    }
    var SYSCALLS = {
      mappings: {},
      buffers: [null, [], []],
      printChar: function (stream, curr) {
        var buffer = SYSCALLS.buffers[stream];
        if (curr === 0 || curr === 10) {
          (stream === 1 ? out : err)(UTF8ArrayToString(buffer, 0));
          buffer.length = 0;
        } else {
          buffer.push(curr);
        }
      },
      varargs: undefined,
      get: function () {
        SYSCALLS.varargs += 4;
        var ret = HEAP32[(SYSCALLS.varargs - 4) >> 2];
        return ret;
      },
      getStr: function (ptr) {
        var ret = UTF8ToString(ptr);
        return ret;
      },
      get64: function (low, high) {
        return low;
      },
    };
    function _environ_get(__environ, environ_buf) {
      var bufSize = 0;
      getEnvStrings().forEach(function (string, i) {
        var ptr = environ_buf + bufSize;
        HEAP32[(__environ + i * 4) >> 2] = ptr;
        writeAsciiToMemory(string, ptr);
        bufSize += string.length + 1;
      });
      return 0;
    }
    function _environ_sizes_get(penviron_count, penviron_buf_size) {
      var strings = getEnvStrings();
      HEAP32[penviron_count >> 2] = strings.length;
      var bufSize = 0;
      strings.forEach(function (string) {
        bufSize += string.length + 1;
      });
      HEAP32[penviron_buf_size >> 2] = bufSize;
      return 0;
    }
    function _exit(status) {
      exit(status);
    }
    function _fd_close(fd) {
      return 0;
    }
    function _fd_seek(fd, offset_low, offset_high, whence, newOffset) {}
    function _fd_write(fd, iov, iovcnt, pnum) {
      var num = 0;
      for (var i = 0; i < iovcnt; i++) {
        var ptr = HEAP32[(iov + i * 8) >> 2];
        var len = HEAP32[(iov + (i * 8 + 4)) >> 2];
        for (var j = 0; j < len; j++) {
          SYSCALLS.printChar(fd, HEAPU8[ptr + j]);
        }
        num += len;
      }
      HEAP32[pnum >> 2] = num;
      return 0;
    }
    function _pthread_create() {
      return 6;
    }
    function _pthread_join() {
      return 28;
    }
    var asmLibraryArg = {
      a: _abort,
      n: _clock_gettime,
      i: _dlopen,
      d: _dlsym,
      h: _emscripten_get_heap_max,
      l: _emscripten_memcpy_big,
      m: _emscripten_resize_heap,
      o: _emscripten_thread_sleep,
      p: _environ_get,
      f: _environ_sizes_get,
      j: _exit,
      g: _fd_close,
      k: _fd_seek,
      b: _fd_write,
      c: _pthread_create,
      e: _pthread_join,
    };
    var asm = createWasm();
    var ___wasm_call_ctors = (Module['___wasm_call_ctors'] = function () {
      return (___wasm_call_ctors = Module['___wasm_call_ctors'] =
        Module['asm']['r']).apply(null, arguments);
    });
    var _getModelBufferMemoryOffset = (Module['_getModelBufferMemoryOffset'] =
      function () {
        return (_getModelBufferMemoryOffset = Module[
          '_getModelBufferMemoryOffset'
        ] =
          Module['asm']['s']).apply(null, arguments);
      });
    var _getInputMemoryOffset = (Module['_getInputMemoryOffset'] = function () {
      return (_getInputMemoryOffset = Module['_getInputMemoryOffset'] =
        Module['asm']['t']).apply(null, arguments);
    });
    var _getInputHeight = (Module['_getInputHeight'] = function () {
      return (_getInputHeight = Module['_getInputHeight'] =
        Module['asm']['u']).apply(null, arguments);
    });
    var _getInputWidth = (Module['_getInputWidth'] = function () {
      return (_getInputWidth = Module['_getInputWidth'] =
        Module['asm']['v']).apply(null, arguments);
    });
    var _getInputChannelCount = (Module['_getInputChannelCount'] = function () {
      return (_getInputChannelCount = Module['_getInputChannelCount'] =
        Module['asm']['w']).apply(null, arguments);
    });
    var _getOutputMemoryOffset = (Module['_getOutputMemoryOffset'] =
      function () {
        return (_getOutputMemoryOffset = Module['_getOutputMemoryOffset'] =
          Module['asm']['x']).apply(null, arguments);
      });
    var _getOutputHeight = (Module['_getOutputHeight'] = function () {
      return (_getOutputHeight = Module['_getOutputHeight'] =
        Module['asm']['y']).apply(null, arguments);
    });
    var _getOutputWidth = (Module['_getOutputWidth'] = function () {
      return (_getOutputWidth = Module['_getOutputWidth'] =
        Module['asm']['z']).apply(null, arguments);
    });
    var _getOutputChannelCount = (Module['_getOutputChannelCount'] =
      function () {
        return (_getOutputChannelCount = Module['_getOutputChannelCount'] =
          Module['asm']['A']).apply(null, arguments);
      });
    var _loadModel = (Module['_loadModel'] = function () {
      return (_loadModel = Module['_loadModel'] = Module['asm']['B']).apply(
        null,
        arguments
      );
    });
    var _runInference = (Module['_runInference'] = function () {
      return (_runInference = Module['_runInference'] =
        Module['asm']['C']).apply(null, arguments);
    });
    var ___errno_location = (Module['___errno_location'] = function () {
      return (___errno_location = Module['___errno_location'] =
        Module['asm']['E']).apply(null, arguments);
    });
    var calledRun;
    function ExitStatus(status) {
      this.name = 'ExitStatus';
      this.message = 'Program terminated with exit(' + status + ')';
      this.status = status;
    }
    dependenciesFulfilled = function runCaller() {
      if (!calledRun) run();
      if (!calledRun) dependenciesFulfilled = runCaller;
    };
    function run(args) {
      args = args || arguments_;
      if (runDependencies > 0) {
        return;
      }
      preRun();
      if (runDependencies > 0) {
        return;
      }
      function doRun() {
        if (calledRun) return;
        calledRun = true;
        Module['calledRun'] = true;
        if (ABORT) return;
        initRuntime();
        preMain();
        readyPromiseResolve(Module);
        if (Module['onRuntimeInitialized']) Module['onRuntimeInitialized']();
        postRun();
      }
      if (Module['setStatus']) {
        Module['setStatus']('Running...');
        setTimeout(function () {
          setTimeout(function () {
            Module['setStatus']('');
          }, 1);
          doRun();
        }, 1);
      } else {
        doRun();
      }
    }
    Module['run'] = run;
    function exit(status, implicit) {
      EXITSTATUS = status;
      if (implicit && keepRuntimeAlive() && status === 0) {
        return;
      }
      if (keepRuntimeAlive()) {
      } else {
        exitRuntime();
        if (Module['onExit']) Module['onExit'](status);
        ABORT = true;
      }
      quit_(status, new ExitStatus(status));
    }
    if (Module['preInit']) {
      if (typeof Module['preInit'] == 'function')
        Module['preInit'] = [Module['preInit']];
      while (Module['preInit'].length > 0) {
        Module['preInit'].pop()();
      }
    }
    run();

    return createTwilioTFLiteSIMDModule.ready;
  };
})();
if (typeof exports === 'object' && typeof module === 'object')
  module.exports = createTFLiteSIMDModule;
else if (typeof define === 'function' && define['amd'])
  define([], function () {
    return createTFLiteSIMDModule;
  });
else if (typeof exports === 'object')
  exports['createTwilioTFLiteSIMDModule'] = createTFLiteSIMDModule;

