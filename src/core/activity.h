#ifndef _SRC_CORE_ACTIVITY_H
#define _SRC_CORE_ACTIVITY_H

#define ROCPROFILER_V1

#ifdef ROCP_INTERNAL_BUILD
#include "inc/rocprofiler.h"
#else
#include <rocprofiler/rocprofiler.h>
#endif

#include <stdint.h>

// HSA EVT ID enumeration
enum hsa_evt_id_t {
  HSA_EVT_ID_ALLOCATE = ROCPROFILER_HSA_CB_ID_ALLOCATE,
  HSA_EVT_ID_DEVICE = ROCPROFILER_HSA_CB_ID_DEVICE,
  HSA_EVT_ID_MEMCOPY = ROCPROFILER_HSA_CB_ID_MEMCOPY,
  HSA_EVT_ID_SUBMIT = ROCPROFILER_HSA_CB_ID_SUBMIT,
  HSA_EVT_ID_KSYMBOL = ROCPROFILER_HSA_CB_ID_KSYMBOL,
  HSA_EVT_ID_CODEOBJ = ROCPROFILER_HSA_CB_ID_CODEOBJ,
  HSA_EVT_ID_NUMBER
};

// HSA EVT callback data type
typedef rocprofiler_hsa_callback_data_t hsa_evt_data_t;

#endif // _SRC_CORE_ACTIVITY_H
