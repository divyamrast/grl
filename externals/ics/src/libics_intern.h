/*
 * libics: Image Cytometry Standard file reading and writing.
 *
 * Copyright (C) 2000-2010 Cris Luengo and others
 * email: clluengo@users.sourceforge.net
 *
 * Large chunks of this library written by
 *    Bert Gijsbers
 *    Dr. Hans T.M. van der Voort
 * And also Damir Sudar, Geert van Kempen, Jan Jitze Krol,
 * Chiel Baarslag and Fons Laan.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the Free
 * Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */
/*
 * FILE : libics_intern.h
 *
 * Only needed to build the library.
 */

#ifndef LIBICS_INTERN_H
#define LIBICS_INTERN_H

#include <stdio.h>

#ifndef LIBICS_H
#include "libics.h"
#endif

#ifndef LIBICS_LL_H
#include "libics_ll.h"
#endif

#include "libics_conf.h"

/*
 * Error management routines: these make the code look a lot clearer!
 */
#define ICSINIT Ics_Error error = IcsErr_Ok
   /* Declare and initialize the error variable */
#define ICSDECL Ics_Error error
   /* Declare the error variable without initializing (to avoid compiler warnings!) */
#define ICSXR(f) error = (f); if (error) return error
   /* Execute function and return if an error occurred. (eXecute and Return) */
#define ICSCX(f) if (!error) error = (f)
   /* Execute the function only if no error is defined. (Conditional eXecution) */
#define ICSXA(f) if (!error) error = (f); else (f)
   /* Execute the function, don't overwrite a possible error condition. (eXecute and Append) */
#define ICSTR(t,e) if (t) return (e)
   /* If t is true, returns with e. (Test and Return) */

/*
 * Macros to assert file mode -- used in top-level interface only
 */
#define ICS_FM_RD(p)  if ((p == NULL) || ((p)->FileMode == IcsFileMode_write)) return IcsErr_NotValidAction
   /* Test to see if reading data is allowed */
#define ICS_FM_WD(p)  if ((p == NULL) || ((p)->FileMode != IcsFileMode_write)) return IcsErr_NotValidAction
   /* Test to see if writing data is allowed */
#define ICS_FM_RMD(p) if  (p == NULL)                                          return IcsErr_NotValidAction
   /* Test to see if reading metadata is allowed */
#define ICS_FM_WMD(p) if ((p == NULL) || ((p)->FileMode == IcsFileMode_read))  return IcsErr_NotValidAction
   /* Test to see if writing metadata is allowed */

/*
 * Forcing the proper locale
 */
#ifdef ICS_FORCE_C_LOCALE
   #include <locale.h>
   #define ICS_INIT_LOCALE    char* Ics_CurrentLocale = 0
   #define ICS_SET_LOCALE     Ics_CurrentLocale = setlocale (LC_ALL, NULL); setlocale (LC_ALL, "C")
   #define ICS_REVERT_LOCALE  setlocale (LC_ALL, Ics_CurrentLocale)
#else
   #define ICS_INIT_LOCALE
   #define ICS_SET_LOCALE
   #define ICS_REVERT_LOCALE
#endif

/*
 * Below are defined the IcsTokens. Each corresponds to an ICS
 * keyword. Several token are defined for intenal bookkeeping:
 * *LASTMAIN, *FIRST*, *LAST*. These should not be moved!
 *
 * Note: If a token is added/removed the corresponding arrays which
 * relate token to strings in libics_data.c MUST be synchronized!
 */
typedef enum {
   /* Main category tokens: */
   ICSTOK_SOURCE = 0,
   ICSTOK_LAYOUT,
   ICSTOK_REPRES,
   ICSTOK_PARAM,
   ICSTOK_HISTORY,
   ICSTOK_SENSOR,
   ICSTOK_END,
   ICSTOK_LASTMAIN,

   /* Subcategory tokens: */
   ICSTOK_FIRSTSUB,
   ICSTOK_FILE,
   ICSTOK_OFFSET,
   ICSTOK_PARAMS,
   ICSTOK_ORDER,
   ICSTOK_SIZES,
   ICSTOK_COORD,
   ICSTOK_SIGBIT,
   ICSTOK_FORMAT,
   ICSTOK_SIGN,
   ICSTOK_COMPR,
   ICSTOK_BYTEO,
   ICSTOK_ORIGIN,
   ICSTOK_SCALE,
   ICSTOK_UNITS,
   ICSTOK_LABELS,
   ICSTOK_SCILT,
   ICSTOK_TYPE,
   ICSTOK_MODEL,
   ICSTOK_SPARAMS,
   ICSTOK_LASTSUB,

   /* SubsubCategory tokens: */
   ICSTOK_FIRSTSUBSUB,
   ICSTOK_CHANS,
   ICSTOK_PINHRAD,
   ICSTOK_LAMBDEX,
   ICSTOK_LAMBDEM,
   ICSTOK_PHOTCNT,
   ICSTOK_REFRIME,
   ICSTOK_NUMAPER,
   ICSTOK_REFRILM,
   ICSTOK_PINHSPA,
   ICSTOK_LASTSUBSUB,

   /* Value tokens: */
   ICSTOK_FIRSTVALUE,
   ICSTOK_COMPR_UNCOMPRESSED,
   ICSTOK_COMPR_COMPRESS,
   ICSTOK_COMPR_GZIP,
   ICSTOK_FORMAT_INTEGER,
   ICSTOK_FORMAT_REAL,
   ICSTOK_FORMAT_COMPLEX,
   ICSTOK_SIGN_SIGNED,
   ICSTOK_SIGN_UNSIGNED,
   ICSTOK_LASTVALUE,

   ICSTOK_NONE
} Ics_Token;

/* Definition keyword relating to imel representation */
#define ICS_ORDER_BITS             "bits"
#define ICS_LABEL_BITS             "intensity"

/* Definition of other keywords */
#define ICS_HISTORY                "history"
#define ICS_COORD_VIDEO            "video"
#define ICS_FILENAME               "filename"
#define ICS_VERSION                "ics_version"
#define ICS_UNITS_RELATIVE         "relative"
#define ICS_UNITS_UNDEFINED        "undefined"

/* The following structure links names to (enumerated) tokens: */
typedef struct {
   char const* Name;
   Ics_Token Token;
} Ics_Symbol;

typedef struct {
   int Entries;
   Ics_Symbol* List;
} Ics_SymbolList;

extern Ics_Symbol G_CatSymbols[];
extern Ics_Symbol G_SubCatSymbols[];
extern Ics_Symbol G_SubSubCatSymbols[];
extern Ics_Symbol G_ValueSymbols[];
extern Ics_SymbolList G_Categories;
extern Ics_SymbolList G_SubCategories;
extern Ics_SymbolList G_SubSubCategories;
extern Ics_SymbolList G_Values;

/* This is the actual stuff behind the "void* History" in the ICS structure: */
typedef struct {
    char** Strings;             /* History strings */
    size_t Length;              /* Size of the Strings array */
    int NStr;                   /* Index past the last one in the array; sort of the
                                   number of strings in the array, except that some
                                   array elements might be NULL */
} Ics_History;

/* This is the actual stuff behind the "void* BlockRead" in the ICS structure: */
typedef struct {
    FILE* DataFilePtr;          /* Input data file */
#ifdef ICS_ZLIB
    void* ZlibStream;           /* z_stream* (or gzFile) for zlib */
    void* ZlibInputBuffer;      /* Input buffer for compressed data */
    unsigned long ZlibCRC;      /* running CRC */
#endif
    int CompressRead;           /* set to non-zero when IcsReadCompress has been called */
} Ics_BlockRead;

/* Assorted support functions */
size_t IcsStrToSize (char const* str);
void IcsStrCpy (char* dest, char const* src, int len);
void IcsAppendChar (char* Line, char ch);
int IcsGetBytesPerSample (Ics_Header const* IcsStruct);
void IcsGetFileName (char* dest, char const* src);

Ics_Error IcsOpenIcs (FILE** fpp, char* filename, int forcename);

Ics_Error IcsInternAddHistory (Ics_Header* ics, char const* key, char const* stuff,
                               char const* seps);

/* Binary data support functions */
void IcsFillByteOrder (int bytes, int machineByteOrder[ICS_MAX_IMEL_SIZE]);
Ics_Error IcsWritePlainWithStrides (void const* src, size_t const* dim, size_t const* stride,
                                    int ndims, int nbytes, FILE* file);
Ics_Error IcsCopyIds (char const* infilename, size_t inoffset, char const* outfilename);

/* zlib interface functions */
Ics_Error IcsWriteZip (void const* src, size_t n, FILE* fp, int CompLevel);
Ics_Error IcsWriteZipWithStrides (void const* src, size_t const* dim, size_t const* stride,
                                  int ndims, int nbytes, FILE* file, int level);
Ics_Error IcsOpenZip (Ics_Header* IcsStruct);
Ics_Error IcsCloseZip (Ics_Header* IcsStruct);
Ics_Error IcsReadZipBlock (Ics_Header* IcsStruct, void* outbuf, size_t len);
Ics_Error IcsSetZipBlock (Ics_Header* IcsStruct, long offset, int whence);

/* Reading COMPRESS-compressed data */
Ics_Error IcsReadCompress (Ics_Header* IcsStruct, void* outbuf, size_t len);

#endif /* LIBICS_INTERN_H */
