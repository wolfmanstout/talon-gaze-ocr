#pragma once
#include "Python.h"
#include <jaro_winkler/jaro_winkler.hpp>
#include <exception>

#include "rapidfuzz_capi.h"

#define PYTHON_VERSION(major, minor, micro) ((major << 24) | (minor << 16) | (micro << 8))

class PythonTypeError: public std::bad_typeid {
public:

    PythonTypeError(char const* error)
      : m_error(error) {}

    virtual char const* what() const noexcept {
        return m_error;
    }
private:
    char const* m_error;
};

/* copy from cython */
static inline void CppExn2PyErr() {
  try {
    if (PyErr_Occurred())
      ; // let the latest Python exn pass through and ignore the current one
    else
      throw;
  } catch (const std::bad_alloc& exn) {
    PyErr_SetString(PyExc_MemoryError, exn.what());
  } catch (const std::bad_cast& exn) {
    PyErr_SetString(PyExc_TypeError, exn.what());
  } catch (const std::bad_typeid& exn) {
    PyErr_SetString(PyExc_TypeError, exn.what());
  } catch (const std::domain_error& exn) {
    PyErr_SetString(PyExc_ValueError, exn.what());
  } catch (const std::invalid_argument& exn) {
    PyErr_SetString(PyExc_ValueError, exn.what());
  } catch (const std::ios_base::failure& exn) {
    PyErr_SetString(PyExc_IOError, exn.what());
  } catch (const std::out_of_range& exn) {
    PyErr_SetString(PyExc_IndexError, exn.what());
  } catch (const std::overflow_error& exn) {
    PyErr_SetString(PyExc_OverflowError, exn.what());
  } catch (const std::range_error& exn) {
    PyErr_SetString(PyExc_ArithmeticError, exn.what());
  } catch (const std::underflow_error& exn) {
    PyErr_SetString(PyExc_ArithmeticError, exn.what());
  } catch (const std::exception& exn) {
    PyErr_SetString(PyExc_RuntimeError, exn.what());
  }
  catch (...)
  {
    PyErr_SetString(PyExc_RuntimeError, "Unknown exception");
  }
}

#define LIST_OF_CASES()   \
    X_ENUM(RF_UINT8,  uint8_t ) \
    X_ENUM(RF_UINT16, uint16_t) \
    X_ENUM(RF_UINT32, uint32_t) \
    X_ENUM(RF_UINT64, uint64_t)

/* RAII Wrapper for RF_String */
struct RF_StringWrapper {
    RF_String string;
    PyObject* obj;

    RF_StringWrapper()
        : string({nullptr, (RF_StringType)0, nullptr, 0, nullptr}), obj(nullptr) {}

    RF_StringWrapper(RF_String string_)
        : string(string_), obj(nullptr) {}

    RF_StringWrapper(RF_String string_, PyObject* o)
        : string(string_), obj(o)
    {
        Py_XINCREF(obj);
    }

    RF_StringWrapper(const RF_StringWrapper&) = delete;
    RF_StringWrapper& operator=(const RF_StringWrapper&) = delete;

    RF_StringWrapper(RF_StringWrapper&& other)
        : RF_StringWrapper()
    {
        swap(*this, other);
    }

    RF_StringWrapper& operator=(RF_StringWrapper&& other) {
        if (&other != this) {
            if (string.dtor) {
                string.dtor(&string);
            }
            Py_XDECREF(obj);
            string = other.string;
            obj = other.obj;
            other.string = {nullptr, (RF_StringType)0, nullptr, 0, nullptr};
            other.obj = nullptr;
      }
      return *this;
    };

    ~RF_StringWrapper() {
        if (string.dtor) {
            string.dtor(&string);
        }
        Py_XDECREF(obj);
    }

    friend void swap(RF_StringWrapper& first, RF_StringWrapper& second) noexcept
    {
        using std::swap;
        swap(first.string, second.string);
        swap(first.obj, second.obj);
    }
};

void default_string_deinit(RF_String* string)
{
    free(string->data);
}

template <typename Func, typename... Args>
auto visit(const RF_String& str, Func&& f, Args&&... args)
{
    switch(str.kind) {
# define X_ENUM(kind, type) case kind: return f((type*)str.data, (type*)str.data + str.length, std::forward<Args>(args)...);
    LIST_OF_CASES()
# undef X_ENUM
    default:
        throw std::logic_error("Invalid string type");
    }
}

template <typename Func, typename... Args>
auto visitor(const RF_String& str1, const RF_String& str2, Func&& f, Args&&... args)
{
    return visit(str2,
        [&](auto first, auto last) {
            return visit(str1, std::forward<Func>(f), first, last, std::forward<Args>(args)...);
        }
    );
}

static inline bool is_valid_string(PyObject* py_str)
{
    bool is_string = false;

    if (PyBytes_Check(py_str)) {
        is_string = true;
    }
    else if (PyUnicode_Check(py_str)) {
        // PEP 623 deprecates legacy strings and therefor
        // deprecates e.g. PyUnicode_READY in Python 3.10
#if PY_VERSION_HEX < PYTHON_VERSION(3, 10, 0)
        if (PyUnicode_READY(py_str)) {
          // cython will use the exception set by PyUnicode_READY
          throw std::runtime_error("");
        }
#endif
        is_string = true;
    }

    return is_string;
}

static inline void validate_string(PyObject* py_str, const char* err)
{
    if (PyBytes_Check(py_str)) {
        return;
    }
    else if (PyUnicode_Check(py_str)) {
        // PEP 623 deprecates legacy strings and therefor
        // deprecates e.g. PyUnicode_READY in Python 3.10
#if PY_VERSION_HEX < PYTHON_VERSION(3, 10, 0)
        if (PyUnicode_READY(py_str)) {
          // cython will use the exception set by PyUnicode_READY
          throw std::runtime_error("");
        }
#endif
        return;
    }

    throw PythonTypeError(err);
}

static inline RF_String convert_string(PyObject* py_str)
{
    if (PyBytes_Check(py_str)) {
        return {
            nullptr,
            RF_UINT8,
            PyBytes_AS_STRING(py_str),
            static_cast<int64_t>(PyBytes_GET_SIZE(py_str)),
            nullptr
        };
    } else {
        RF_StringType kind;
        switch(PyUnicode_KIND(py_str)) {
        case PyUnicode_1BYTE_KIND:
           kind = RF_UINT8;
           break;
        case PyUnicode_2BYTE_KIND:
           kind = RF_UINT16;
           break;
        default:
           kind = RF_UINT32;
           break;
        }

        return {
            nullptr,
            kind,
            PyUnicode_DATA(py_str),
            static_cast<int64_t>(PyUnicode_GET_LENGTH(py_str)),
            nullptr
        };
    }
}

template <typename CachedScorer>
static void scorer_deinit(RF_ScorerFunc* self)
{
    delete (CachedScorer*)self->context;
}

template<typename CachedScorer>
static inline bool scorer_func_wrapper_f64(const RF_ScorerFunc* self, const RF_String* str, int64_t str_count, double score_cutoff, double* result)
{
    CachedScorer& scorer = *(CachedScorer*)self->context;
    try {
        if (str_count != 1)
        {
            throw std::logic_error("Only str_count == 1 supported");
        }
        *result = visit(*str, [&](auto first, auto last){
            return scorer.similarity(first, last, score_cutoff);
        });
    } catch(...) {
      PyGILState_STATE gilstate_save = PyGILState_Ensure();
      CppExn2PyErr();
      PyGILState_Release(gilstate_save);
      return false;
    }
    return true;
}

template<template <typename> class CachedScorer, typename InputIt1, typename ...Args>
static inline RF_ScorerFunc get_ScorerContext_f64(InputIt1 first1, InputIt1 last1, Args... args)
{
    using CharT1 = typename std::iterator_traits<InputIt1>::value_type;
    RF_ScorerFunc context;
    context.context = (void*) new CachedScorer<CharT1>(first1, last1, args...);

    context.call.f64 = scorer_func_wrapper_f64<CachedScorer<CharT1>>;
    context.dtor = scorer_deinit<CachedScorer<CharT1>>;
    return context;
}

template<template <typename> class CachedScorer, typename ...Args>
static inline bool scorer_init_f64(RF_ScorerFunc* self, int64_t str_count, const RF_String* strings, Args... args)
{
    try {
        if (str_count != 1)
        {
            throw std::logic_error("Only str_count == 1 supported");
        }
        *self = visit(*strings, [&](auto first, auto last){
            return get_ScorerContext_f64<CachedScorer>(first, last, args...);
        });
    } catch(...) {
      PyGILState_STATE gilstate_save = PyGILState_Ensure();
      CppExn2PyErr();
      PyGILState_Release(gilstate_save);
      return false;
    }
    return true;
}
