/*

Copyright (c) 2023, PyData Development Team
All rights reserved.

Distributed under the terms of the BSD Simplified License.

*/
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "pandas/parser/tokenizer.h"

typedef struct {
  int (*to_double)(char *, double *, char, char, int *);
  int (*floatify)(PyObject *, double *, int *);
  void *(*new_rd_source)(PyObject *);
  int (*del_rd_source)(void *);
  void *(*buffer_rd_bytes)(void *, size_t, size_t *, int *, const char *);
  void (*uint_state_init)(uint_state *);
  int (*uint64_conflict)(uint_state *);
  void (*coliter_setup)(coliter_t *, parser_t *, int64_t, int64_t);
  parser_t *(*parser_new)(void);
  int (*parser_init)(parser_t *);
  void (*parser_free)(parser_t *);
  void (*parser_del)(parser_t *);
  int (*parser_add_skiprow)(parser_t *, int64_t);
  int (*parser_set_skipfirstnrows)(parser_t *, int64_t);
  void (*parser_set_default_options)(parser_t *);
  int (*parser_consume_rows)(parser_t *, size_t);
  int (*parser_trim_buffers)(parser_t *);
  int (*tokenize_all_rows)(parser_t *, const char *);
  int (*tokenize_nrows)(parser_t *, size_t, const char *);
  int64_t (*str_to_int64)(const char *, int64_t, int64_t, int *, char);
  uint64_t (*str_to_uint64)(uint_state *, const char *, int64_t, uint64_t,
                            int *, char);
  double (*xstrtod)(const char *, char **, char, char, char, int, int *, int *);
  double (*precise_xstrtod)(const char *, char **, char, char, char, int, int *,
                            int *);
  double (*round_trip)(const char *, char **, char, char, char, int, int *,
                       int *);
  int (*to_boolean)(const char *, uint8_t *);
} PandasParser_CAPI;

#define PandasParser_CAPSULE_NAME "pandas._pandas_parser_CAPI"

#ifndef _PANDAS_PARSER_IMPL
static PandasParser_CAPI *PandasParserAPI = NULL;

#define PandasParser_IMPORT                                                    \
  PandasParserAPI =                                                            \
      (PandasParser_CAPI *)PyCapsule_Import(PandasParser_CAPSULE_NAME, 0)

#define to_double(item, p_value, sci, decimal, maybe_int)                      \
  PandasParserAPI->to_double((item), (p_value), (sci), (decimal), (maybe_int))
#define floatify(str, result, maybe_int)                                       \
  PandasParserAPI->floatify((str), (result), (maybe_int))
#define new_rd_source(obj) PandasParserAPI->new_rd_source((obj))
#define del_rd_source(src) PandasParserAPI->del_rd_source((src))
#define buffer_rd_bytes(source, nbytes, bytes_read, status, encoding_errors)   \
  PandasParserAPI->buffer_rd_bytes((source), (nbytes), (bytes_read), (status), \
                                   (encoding_errors))
#define uint_state_init(self) PandasParserAPI->uint_state_init((self))
#define uint64_conflict(self) PandasParserAPI->uint64_conflict((self))
#define coliter_setup(self, parser, i, start)                                  \
  PandasParserAPI->coliter_setup((self), (parser), (i), (start))
#define parser_new PandasParserAPI->parser_new
#define parser_init(self) PandasParserAPI->parser_init((self))
#define parser_free(self) PandasParserAPI->parser_free((self))
#define parser_del(self) PandasParserAPI->parser_del((self))
#define parser_add_skiprow(self, row)                                          \
  PandasParserAPI->parser_add_skiprow((self), (row))
#define parser_set_skipfirstnrows(self, nrows)                                 \
  PandasParserAPI->parser_set_skipfirstnrows((self), (nrows))
#define parser_set_default_options(self)                                       \
  PandasParserAPI->parser_set_default_options((self))
#define parser_consume_rows(self, nrows)                                       \
  PandasParserAPI->parser_consume_rows((self), (nrows))
#define parser_trim_buffers(self)                                              \
  PandasParserAPI->parser_trim_buffers((self))
#define tokenize_all_rows(self, encoding_errors)                        \
  PandasParserAPI->tokenize_all_rows((self), (encoding_errors))
#define tokenize_nrows(self, nrows, encoding_errors)                    \
  PandasParserAPI->tokenize_nrows((self), (nrows), (encoding_errors))
#define str_to_int64(p_item, int_min, int_max, error, t_sep)                   \
  PandasParserAPI->str_to_int64((p_item), (int_min), (int_max), (error),       \
                                (t_sep))
#define str_to_uint64(state, p_item, int_max, uint_max, error, t_sep)          \
  PandasParserAPI->str_to_uint64((state), (p_item), (int_max), (uint_max),     \
                                 (error), (t_sep))
#define xstrtod(p, q, decimal, sci, tsep, skip_trailing, error, maybe_int)     \
  PandasParserAPI->xstrtod((p), (q), (decimal), (sci), (tsep),                 \
                           (skip_trailing), (error), (maybe_int))
#define precise_xstrtod(p, q, decimal, sci, tsep, skip_trailing, error,        \
                        maybe_int)                                             \
  PandasParserAPI->precise_xstrtod((p), (q), (decimal), (sci), (tsep),         \
                                   (skip_trailing), (error), (maybe_int))
#define round_trip(p, q, decimal, sci, tsep, skip_trailing, error, maybe_int)  \
  PandasParserAPI->round_trip((p), (q), (decimal), (sci), (tsep),              \
                              (skip_trailing), (error), (maybe_int))
#define to_boolean(item, val) PandasParserAPI->to_boolean((item), (val))
#endif  /* !defined(_PANDAS_PARSER_IMPL) */

#ifdef __cplusplus
}
#endif
