/* Header file automatically generated from Microsoft.AI.MachineLearning.idl */
/*
 * File built with Microsoft(R) MIDLRT Compiler Engine Version 10.00.0229 
 */

#pragma warning( disable: 4049 )  /* more than 64k source lines */

/* verify that the <rpcndr.h> version is high enough to compile this file*/
#ifndef __REQUIRED_RPCNDR_H_VERSION__
#define __REQUIRED_RPCNDR_H_VERSION__ 500
#endif

/* verify that the <rpcsal.h> version is high enough to compile this file*/
#ifndef __REQUIRED_RPCSAL_H_VERSION__
#define __REQUIRED_RPCSAL_H_VERSION__ 100
#endif

#include <rpc.h>
#include <rpcndr.h>

#ifndef __RPCNDR_H_VERSION__
#error this stub requires an updated version of <rpcndr.h>
#endif /* __RPCNDR_H_VERSION__ */

#ifndef COM_NO_WINDOWS_H
#include <windows.h>
#include <ole2.h>
#endif /*COM_NO_WINDOWS_H*/
#ifndef __Microsoft2EAI2EMachineLearning_h__
#define __Microsoft2EAI2EMachineLearning_h__
#ifndef __Microsoft2EAI2EMachineLearning_p_h__
#define __Microsoft2EAI2EMachineLearning_p_h__


#pragma once

// Ensure that the setting of the /ns_prefix command line switch is consistent for all headers.
// If you get an error from the compiler indicating "warning C4005: 'CHECK_NS_PREFIX_STATE': macro redefinition", this
// indicates that you have included two different headers with different settings for the /ns_prefix MIDL command line switch
#if !defined(DISABLE_NS_PREFIX_CHECKS)
#define CHECK_NS_PREFIX_STATE "always"
#endif // !defined(DISABLE_NS_PREFIX_CHECKS)


#pragma push_macro("MIDL_CONST_ID")
#undef MIDL_CONST_ID
#define MIDL_CONST_ID const __declspec(selectany)


//  API Contract Inclusion Definitions
#if !defined(SPECIFIC_API_CONTRACT_DEFINITIONS)
#if !defined(MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION)
#define MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION 0x50000
#endif // defined(MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION)

#if !defined(WINDOWS_APPLICATIONMODEL_ACTIVATION_ACTIVATEDEVENTSCONTRACT_VERSION)
#define WINDOWS_APPLICATIONMODEL_ACTIVATION_ACTIVATEDEVENTSCONTRACT_VERSION 0x10000
#endif // defined(WINDOWS_APPLICATIONMODEL_ACTIVATION_ACTIVATEDEVENTSCONTRACT_VERSION)

#if !defined(WINDOWS_APPLICATIONMODEL_ACTIVATION_ACTIVATIONCAMERASETTINGSCONTRACT_VERSION)
#define WINDOWS_APPLICATIONMODEL_ACTIVATION_ACTIVATIONCAMERASETTINGSCONTRACT_VERSION 0x10000
#endif // defined(WINDOWS_APPLICATIONMODEL_ACTIVATION_ACTIVATIONCAMERASETTINGSCONTRACT_VERSION)

#if !defined(WINDOWS_APPLICATIONMODEL_ACTIVATION_CONTACTACTIVATEDEVENTSCONTRACT_VERSION)
#define WINDOWS_APPLICATIONMODEL_ACTIVATION_CONTACTACTIVATEDEVENTSCONTRACT_VERSION 0x10000
#endif // defined(WINDOWS_APPLICATIONMODEL_ACTIVATION_CONTACTACTIVATEDEVENTSCONTRACT_VERSION)

#if !defined(WINDOWS_APPLICATIONMODEL_ACTIVATION_WEBUISEARCHACTIVATEDEVENTSCONTRACT_VERSION)
#define WINDOWS_APPLICATIONMODEL_ACTIVATION_WEBUISEARCHACTIVATEDEVENTSCONTRACT_VERSION 0x10000
#endif // defined(WINDOWS_APPLICATIONMODEL_ACTIVATION_WEBUISEARCHACTIVATEDEVENTSCONTRACT_VERSION)

#if !defined(WINDOWS_APPLICATIONMODEL_BACKGROUND_BACKGROUNDALARMAPPLICATIONCONTRACT_VERSION)
#define WINDOWS_APPLICATIONMODEL_BACKGROUND_BACKGROUNDALARMAPPLICATIONCONTRACT_VERSION 0x10000
#endif // defined(WINDOWS_APPLICATIONMODEL_BACKGROUND_BACKGROUNDALARMAPPLICATIONCONTRACT_VERSION)

#if !defined(WINDOWS_APPLICATIONMODEL_CALLS_BACKGROUND_CALLSBACKGROUNDCONTRACT_VERSION)
#define WINDOWS_APPLICATIONMODEL_CALLS_BACKGROUND_CALLSBACKGROUNDCONTRACT_VERSION 0x20000
#endif // defined(WINDOWS_APPLICATIONMODEL_CALLS_BACKGROUND_CALLSBACKGROUNDCONTRACT_VERSION)

#if !defined(WINDOWS_APPLICATIONMODEL_CALLS_CALLSPHONECONTRACT_VERSION)
#define WINDOWS_APPLICATIONMODEL_CALLS_CALLSPHONECONTRACT_VERSION 0x50000
#endif // defined(WINDOWS_APPLICATIONMODEL_CALLS_CALLSPHONECONTRACT_VERSION)

#if !defined(WINDOWS_APPLICATIONMODEL_CALLS_CALLSVOIPCONTRACT_VERSION)
#define WINDOWS_APPLICATIONMODEL_CALLS_CALLSVOIPCONTRACT_VERSION 0x40000
#endif // defined(WINDOWS_APPLICATIONMODEL_CALLS_CALLSVOIPCONTRACT_VERSION)

#if !defined(WINDOWS_APPLICATIONMODEL_CALLS_LOCKSCREENCALLCONTRACT_VERSION)
#define WINDOWS_APPLICATIONMODEL_CALLS_LOCKSCREENCALLCONTRACT_VERSION 0x10000
#endif // defined(WINDOWS_APPLICATIONMODEL_CALLS_LOCKSCREENCALLCONTRACT_VERSION)

#if !defined(WINDOWS_APPLICATIONMODEL_COMMUNICATIONBLOCKING_COMMUNICATIONBLOCKINGCONTRACT_VERSION)
#define WINDOWS_APPLICATIONMODEL_COMMUNICATIONBLOCKING_COMMUNICATIONBLOCKINGCONTRACT_VERSION 0x20000
#endif // defined(WINDOWS_APPLICATIONMODEL_COMMUNICATIONBLOCKING_COMMUNICATIONBLOCKINGCONTRACT_VERSION)

#if !defined(WINDOWS_APPLICATIONMODEL_FULLTRUSTAPPCONTRACT_VERSION)
#define WINDOWS_APPLICATIONMODEL_FULLTRUSTAPPCONTRACT_VERSION 0x10000
#endif // defined(WINDOWS_APPLICATIONMODEL_FULLTRUSTAPPCONTRACT_VERSION)

#if !defined(WINDOWS_APPLICATIONMODEL_SEARCH_SEARCHCONTRACT_VERSION)
#define WINDOWS_APPLICATIONMODEL_SEARCH_SEARCHCONTRACT_VERSION 0x10000
#endif // defined(WINDOWS_APPLICATIONMODEL_SEARCH_SEARCHCONTRACT_VERSION)

#if !defined(WINDOWS_APPLICATIONMODEL_STARTUPTASKCONTRACT_VERSION)
#define WINDOWS_APPLICATIONMODEL_STARTUPTASKCONTRACT_VERSION 0x30000
#endif // defined(WINDOWS_APPLICATIONMODEL_STARTUPTASKCONTRACT_VERSION)

#if !defined(WINDOWS_APPLICATIONMODEL_WALLET_WALLETCONTRACT_VERSION)
#define WINDOWS_APPLICATIONMODEL_WALLET_WALLETCONTRACT_VERSION 0x10000
#endif // defined(WINDOWS_APPLICATIONMODEL_WALLET_WALLETCONTRACT_VERSION)

#if !defined(WINDOWS_DEVICES_PRINTERS_EXTENSIONS_EXTENSIONSCONTRACT_VERSION)
#define WINDOWS_DEVICES_PRINTERS_EXTENSIONS_EXTENSIONSCONTRACT_VERSION 0x20000
#endif // defined(WINDOWS_DEVICES_PRINTERS_EXTENSIONS_EXTENSIONSCONTRACT_VERSION)

#if !defined(WINDOWS_DEVICES_SMARTCARDS_SMARTCARDBACKGROUNDTRIGGERCONTRACT_VERSION)
#define WINDOWS_DEVICES_SMARTCARDS_SMARTCARDBACKGROUNDTRIGGERCONTRACT_VERSION 0x30000
#endif // defined(WINDOWS_DEVICES_SMARTCARDS_SMARTCARDBACKGROUNDTRIGGERCONTRACT_VERSION)

#if !defined(WINDOWS_DEVICES_SMARTCARDS_SMARTCARDEMULATORCONTRACT_VERSION)
#define WINDOWS_DEVICES_SMARTCARDS_SMARTCARDEMULATORCONTRACT_VERSION 0x60000
#endif // defined(WINDOWS_DEVICES_SMARTCARDS_SMARTCARDEMULATORCONTRACT_VERSION)

#if !defined(WINDOWS_DEVICES_SMS_LEGACYSMSAPICONTRACT_VERSION)
#define WINDOWS_DEVICES_SMS_LEGACYSMSAPICONTRACT_VERSION 0x10000
#endif // defined(WINDOWS_DEVICES_SMS_LEGACYSMSAPICONTRACT_VERSION)

#if !defined(WINDOWS_FOUNDATION_FOUNDATIONCONTRACT_VERSION)
#define WINDOWS_FOUNDATION_FOUNDATIONCONTRACT_VERSION 0x40000
#endif // defined(WINDOWS_FOUNDATION_FOUNDATIONCONTRACT_VERSION)

#if !defined(WINDOWS_FOUNDATION_UNIVERSALAPICONTRACT_VERSION)
#define WINDOWS_FOUNDATION_UNIVERSALAPICONTRACT_VERSION 0xa0000
#endif // defined(WINDOWS_FOUNDATION_UNIVERSALAPICONTRACT_VERSION)

#if !defined(WINDOWS_GAMING_INPUT_GAMINGINPUTPREVIEWCONTRACT_VERSION)
#define WINDOWS_GAMING_INPUT_GAMINGINPUTPREVIEWCONTRACT_VERSION 0x10000
#endif // defined(WINDOWS_GAMING_INPUT_GAMINGINPUTPREVIEWCONTRACT_VERSION)

#if !defined(WINDOWS_GLOBALIZATION_GLOBALIZATIONJAPANESEPHONETICANALYZERCONTRACT_VERSION)
#define WINDOWS_GLOBALIZATION_GLOBALIZATIONJAPANESEPHONETICANALYZERCONTRACT_VERSION 0x10000
#endif // defined(WINDOWS_GLOBALIZATION_GLOBALIZATIONJAPANESEPHONETICANALYZERCONTRACT_VERSION)

#if !defined(WINDOWS_MEDIA_CAPTURE_APPBROADCASTCONTRACT_VERSION)
#define WINDOWS_MEDIA_CAPTURE_APPBROADCASTCONTRACT_VERSION 0x20000
#endif // defined(WINDOWS_MEDIA_CAPTURE_APPBROADCASTCONTRACT_VERSION)

#if !defined(WINDOWS_MEDIA_CAPTURE_APPCAPTURECONTRACT_VERSION)
#define WINDOWS_MEDIA_CAPTURE_APPCAPTURECONTRACT_VERSION 0x40000
#endif // defined(WINDOWS_MEDIA_CAPTURE_APPCAPTURECONTRACT_VERSION)

#if !defined(WINDOWS_MEDIA_CAPTURE_APPCAPTUREMETADATACONTRACT_VERSION)
#define WINDOWS_MEDIA_CAPTURE_APPCAPTUREMETADATACONTRACT_VERSION 0x10000
#endif // defined(WINDOWS_MEDIA_CAPTURE_APPCAPTUREMETADATACONTRACT_VERSION)

#if !defined(WINDOWS_MEDIA_CAPTURE_CAMERACAPTUREUICONTRACT_VERSION)
#define WINDOWS_MEDIA_CAPTURE_CAMERACAPTUREUICONTRACT_VERSION 0x10000
#endif // defined(WINDOWS_MEDIA_CAPTURE_CAMERACAPTUREUICONTRACT_VERSION)

#if !defined(WINDOWS_MEDIA_CAPTURE_GAMEBARCONTRACT_VERSION)
#define WINDOWS_MEDIA_CAPTURE_GAMEBARCONTRACT_VERSION 0x10000
#endif // defined(WINDOWS_MEDIA_CAPTURE_GAMEBARCONTRACT_VERSION)

#if !defined(WINDOWS_MEDIA_DEVICES_CALLCONTROLCONTRACT_VERSION)
#define WINDOWS_MEDIA_DEVICES_CALLCONTROLCONTRACT_VERSION 0x10000
#endif // defined(WINDOWS_MEDIA_DEVICES_CALLCONTROLCONTRACT_VERSION)

#if !defined(WINDOWS_MEDIA_MEDIACONTROLCONTRACT_VERSION)
#define WINDOWS_MEDIA_MEDIACONTROLCONTRACT_VERSION 0x10000
#endif // defined(WINDOWS_MEDIA_MEDIACONTROLCONTRACT_VERSION)

#if !defined(WINDOWS_MEDIA_PROTECTION_PROTECTIONRENEWALCONTRACT_VERSION)
#define WINDOWS_MEDIA_PROTECTION_PROTECTIONRENEWALCONTRACT_VERSION 0x10000
#endif // defined(WINDOWS_MEDIA_PROTECTION_PROTECTIONRENEWALCONTRACT_VERSION)

#if !defined(WINDOWS_NETWORKING_CONNECTIVITY_WWANCONTRACT_VERSION)
#define WINDOWS_NETWORKING_CONNECTIVITY_WWANCONTRACT_VERSION 0x20000
#endif // defined(WINDOWS_NETWORKING_CONNECTIVITY_WWANCONTRACT_VERSION)

#if !defined(WINDOWS_NETWORKING_SOCKETS_CONTROLCHANNELTRIGGERCONTRACT_VERSION)
#define WINDOWS_NETWORKING_SOCKETS_CONTROLCHANNELTRIGGERCONTRACT_VERSION 0x30000
#endif // defined(WINDOWS_NETWORKING_SOCKETS_CONTROLCHANNELTRIGGERCONTRACT_VERSION)

#if !defined(WINDOWS_PHONE_PHONECONTRACT_VERSION)
#define WINDOWS_PHONE_PHONECONTRACT_VERSION 0x10000
#endif // defined(WINDOWS_PHONE_PHONECONTRACT_VERSION)

#if !defined(WINDOWS_PHONE_PHONEINTERNALCONTRACT_VERSION)
#define WINDOWS_PHONE_PHONEINTERNALCONTRACT_VERSION 0x10000
#endif // defined(WINDOWS_PHONE_PHONEINTERNALCONTRACT_VERSION)

#if !defined(WINDOWS_SECURITY_ENTERPRISEDATA_ENTERPRISEDATACONTRACT_VERSION)
#define WINDOWS_SECURITY_ENTERPRISEDATA_ENTERPRISEDATACONTRACT_VERSION 0x50000
#endif // defined(WINDOWS_SECURITY_ENTERPRISEDATA_ENTERPRISEDATACONTRACT_VERSION)

#if !defined(WINDOWS_STORAGE_PROVIDER_CLOUDFILESCONTRACT_VERSION)
#define WINDOWS_STORAGE_PROVIDER_CLOUDFILESCONTRACT_VERSION 0x40000
#endif // defined(WINDOWS_STORAGE_PROVIDER_CLOUDFILESCONTRACT_VERSION)

#if !defined(WINDOWS_SYSTEM_SYSTEMMANAGEMENTCONTRACT_VERSION)
#define WINDOWS_SYSTEM_SYSTEMMANAGEMENTCONTRACT_VERSION 0x70000
#endif // defined(WINDOWS_SYSTEM_SYSTEMMANAGEMENTCONTRACT_VERSION)

#if !defined(WINDOWS_UI_CORE_COREWINDOWDIALOGSCONTRACT_VERSION)
#define WINDOWS_UI_CORE_COREWINDOWDIALOGSCONTRACT_VERSION 0x10000
#endif // defined(WINDOWS_UI_CORE_COREWINDOWDIALOGSCONTRACT_VERSION)

#if !defined(WINDOWS_UI_VIEWMANAGEMENT_VIEWMANAGEMENTVIEWSCALINGCONTRACT_VERSION)
#define WINDOWS_UI_VIEWMANAGEMENT_VIEWMANAGEMENTVIEWSCALINGCONTRACT_VERSION 0x10000
#endif // defined(WINDOWS_UI_VIEWMANAGEMENT_VIEWMANAGEMENTVIEWSCALINGCONTRACT_VERSION)

#if !defined(WINDOWS_UI_WEBUI_CORE_WEBUICOMMANDBARCONTRACT_VERSION)
#define WINDOWS_UI_WEBUI_CORE_WEBUICOMMANDBARCONTRACT_VERSION 0x10000
#endif // defined(WINDOWS_UI_WEBUI_CORE_WEBUICOMMANDBARCONTRACT_VERSION)

#endif // defined(SPECIFIC_API_CONTRACT_DEFINITIONS)


// Header files for imported files
#include "Windows.Foundation.h"
#include "windows.graphics.h"
#include "windows.media.h"
#include "dualapipartitionattribute.h"
#include "windows.graphics.directx.direct3d11.h"
#include "windows.graphics.imaging.h"
#include "windows.storage.h"
// Importing Collections header
#include <windows.foundation.collections.h>

#if defined(__cplusplus) && !defined(CINTERFACE)
/* Forward Declarations */
#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ILearningModelFeatureDescriptor;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor ABI::Microsoft::AI::MachineLearning::ILearningModelFeatureDescriptor

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelOperatorProvider_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelOperatorProvider_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ILearningModelOperatorProvider;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelOperatorProvider ABI::Microsoft::AI::MachineLearning::ILearningModelOperatorProvider

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelOperatorProvider_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ILearningModel;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel ABI::Microsoft::AI::MachineLearning::ILearningModel

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ILearningModelStatics;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics ABI::Microsoft::AI::MachineLearning::ILearningModelStatics

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDevice_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDevice_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ILearningModelDevice;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDevice ABI::Microsoft::AI::MachineLearning::ILearningModelDevice

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDevice_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceStatics_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceStatics_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ILearningModelDeviceStatics;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceStatics ABI::Microsoft::AI::MachineLearning::ILearningModelDeviceStatics

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceStatics_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceFactory_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceFactory_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ILearningModelDeviceFactory;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceFactory ABI::Microsoft::AI::MachineLearning::ILearningModelDeviceFactory

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceFactory_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelEvaluationResult_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelEvaluationResult_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ILearningModelEvaluationResult;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelEvaluationResult ABI::Microsoft::AI::MachineLearning::ILearningModelEvaluationResult

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelEvaluationResult_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ILearningModelSessionOptions;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions ABI::Microsoft::AI::MachineLearning::ILearningModelSessionOptions

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions2_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions2_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ILearningModelSessionOptions2;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions2 ABI::Microsoft::AI::MachineLearning::ILearningModelSessionOptions2

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions2_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions3_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions3_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ILearningModelSessionOptions3;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions3 ABI::Microsoft::AI::MachineLearning::ILearningModelSessionOptions3

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions3_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ILearningModelSession;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession ABI::Microsoft::AI::MachineLearning::ILearningModelSession

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ILearningModelSessionFactory;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory ABI::Microsoft::AI::MachineLearning::ILearningModelSessionFactory

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory2_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory2_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ILearningModelSessionFactory2;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory2 ABI::Microsoft::AI::MachineLearning::ILearningModelSessionFactory2

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory2_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureValue_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureValue_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ILearningModelFeatureValue;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureValue ABI::Microsoft::AI::MachineLearning::ILearningModelFeatureValue

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureValue_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBinding_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBinding_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ILearningModelBinding;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBinding ABI::Microsoft::AI::MachineLearning::ILearningModelBinding

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBinding_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBindingFactory_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBindingFactory_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ILearningModelBindingFactory;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBindingFactory ABI::Microsoft::AI::MachineLearning::ILearningModelBindingFactory

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBindingFactory_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CIMapFeatureDescriptor_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CIMapFeatureDescriptor_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface IMapFeatureDescriptor;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CIMapFeatureDescriptor ABI::Microsoft::AI::MachineLearning::IMapFeatureDescriptor

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CIMapFeatureDescriptor_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CISequenceFeatureDescriptor_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CISequenceFeatureDescriptor_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ISequenceFeatureDescriptor;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CISequenceFeatureDescriptor ABI::Microsoft::AI::MachineLearning::ISequenceFeatureDescriptor

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CISequenceFeatureDescriptor_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFeatureDescriptor_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFeatureDescriptor_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ITensorFeatureDescriptor;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFeatureDescriptor ABI::Microsoft::AI::MachineLearning::ITensorFeatureDescriptor

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFeatureDescriptor_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface IImageFeatureDescriptor;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor ABI::Microsoft::AI::MachineLearning::IImageFeatureDescriptor

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor2_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor2_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface IImageFeatureDescriptor2;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor2 ABI::Microsoft::AI::MachineLearning::IImageFeatureDescriptor2

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor2_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensor_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensor_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ITensor;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensor ABI::Microsoft::AI::MachineLearning::ITensor

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensor_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ITensorFloat;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat ABI::Microsoft::AI::MachineLearning::ITensorFloat

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ITensorFloatStatics;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics ABI::Microsoft::AI::MachineLearning::ITensorFloatStatics

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics2_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics2_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ITensorFloatStatics2;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics2 ABI::Microsoft::AI::MachineLearning::ITensorFloatStatics2

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics2_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16Bit_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16Bit_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ITensorFloat16Bit;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16Bit ABI::Microsoft::AI::MachineLearning::ITensorFloat16Bit

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16Bit_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ITensorFloat16BitStatics;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics ABI::Microsoft::AI::MachineLearning::ITensorFloat16BitStatics

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics2_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics2_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ITensorFloat16BitStatics2;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics2 ABI::Microsoft::AI::MachineLearning::ITensorFloat16BitStatics2

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics2_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8Bit_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8Bit_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ITensorUInt8Bit;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8Bit ABI::Microsoft::AI::MachineLearning::ITensorUInt8Bit

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8Bit_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ITensorUInt8BitStatics;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics ABI::Microsoft::AI::MachineLearning::ITensorUInt8BitStatics

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics2_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics2_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ITensorUInt8BitStatics2;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics2 ABI::Microsoft::AI::MachineLearning::ITensorUInt8BitStatics2

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics2_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8Bit_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8Bit_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ITensorInt8Bit;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8Bit ABI::Microsoft::AI::MachineLearning::ITensorInt8Bit

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8Bit_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ITensorInt8BitStatics;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics ABI::Microsoft::AI::MachineLearning::ITensorInt8BitStatics

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics2_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics2_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ITensorInt8BitStatics2;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics2 ABI::Microsoft::AI::MachineLearning::ITensorInt8BitStatics2

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics2_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16Bit_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16Bit_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ITensorUInt16Bit;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16Bit ABI::Microsoft::AI::MachineLearning::ITensorUInt16Bit

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16Bit_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ITensorUInt16BitStatics;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics ABI::Microsoft::AI::MachineLearning::ITensorUInt16BitStatics

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics2_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics2_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ITensorUInt16BitStatics2;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics2 ABI::Microsoft::AI::MachineLearning::ITensorUInt16BitStatics2

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics2_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16Bit_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16Bit_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ITensorInt16Bit;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16Bit ABI::Microsoft::AI::MachineLearning::ITensorInt16Bit

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16Bit_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ITensorInt16BitStatics;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics ABI::Microsoft::AI::MachineLearning::ITensorInt16BitStatics

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics2_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics2_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ITensorInt16BitStatics2;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics2 ABI::Microsoft::AI::MachineLearning::ITensorInt16BitStatics2

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics2_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32Bit_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32Bit_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ITensorUInt32Bit;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32Bit ABI::Microsoft::AI::MachineLearning::ITensorUInt32Bit

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32Bit_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ITensorUInt32BitStatics;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics ABI::Microsoft::AI::MachineLearning::ITensorUInt32BitStatics

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics2_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics2_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ITensorUInt32BitStatics2;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics2 ABI::Microsoft::AI::MachineLearning::ITensorUInt32BitStatics2

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics2_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32Bit_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32Bit_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ITensorInt32Bit;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32Bit ABI::Microsoft::AI::MachineLearning::ITensorInt32Bit

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32Bit_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ITensorInt32BitStatics;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics ABI::Microsoft::AI::MachineLearning::ITensorInt32BitStatics

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics2_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics2_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ITensorInt32BitStatics2;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics2 ABI::Microsoft::AI::MachineLearning::ITensorInt32BitStatics2

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics2_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64Bit_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64Bit_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ITensorUInt64Bit;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64Bit ABI::Microsoft::AI::MachineLearning::ITensorUInt64Bit

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64Bit_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ITensorUInt64BitStatics;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics ABI::Microsoft::AI::MachineLearning::ITensorUInt64BitStatics

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics2_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics2_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ITensorUInt64BitStatics2;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics2 ABI::Microsoft::AI::MachineLearning::ITensorUInt64BitStatics2

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics2_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64Bit_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64Bit_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ITensorInt64Bit;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64Bit ABI::Microsoft::AI::MachineLearning::ITensorInt64Bit

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64Bit_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ITensorInt64BitStatics;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics ABI::Microsoft::AI::MachineLearning::ITensorInt64BitStatics

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics2_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics2_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ITensorInt64BitStatics2;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics2 ABI::Microsoft::AI::MachineLearning::ITensorInt64BitStatics2

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics2_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBoolean_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBoolean_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ITensorBoolean;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBoolean ABI::Microsoft::AI::MachineLearning::ITensorBoolean

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBoolean_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ITensorBooleanStatics;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics ABI::Microsoft::AI::MachineLearning::ITensorBooleanStatics

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics2_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics2_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ITensorBooleanStatics2;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics2 ABI::Microsoft::AI::MachineLearning::ITensorBooleanStatics2

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics2_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDouble_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDouble_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ITensorDouble;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDouble ABI::Microsoft::AI::MachineLearning::ITensorDouble

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDouble_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ITensorDoubleStatics;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics ABI::Microsoft::AI::MachineLearning::ITensorDoubleStatics

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics2_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics2_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ITensorDoubleStatics2;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics2 ABI::Microsoft::AI::MachineLearning::ITensorDoubleStatics2

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics2_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorString_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorString_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ITensorString;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorString ABI::Microsoft::AI::MachineLearning::ITensorString

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorString_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ITensorStringStatics;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics ABI::Microsoft::AI::MachineLearning::ITensorStringStatics

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics2_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics2_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface ITensorStringStatics2;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics2 ABI::Microsoft::AI::MachineLearning::ITensorStringStatics2

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics2_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValue_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValue_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface IImageFeatureValue;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValue ABI::Microsoft::AI::MachineLearning::IImageFeatureValue

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValue_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValueStatics_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValueStatics_FWD_DEFINED__
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                interface IImageFeatureValueStatics;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValueStatics ABI::Microsoft::AI::MachineLearning::IImageFeatureValueStatics

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValueStatics_FWD_DEFINED__

// Parameterized interface forward declarations (C++)

// Collection interface definitions

#ifndef DEF___FIKeyValuePair_2_HSTRING_HSTRING_USE
#define DEF___FIKeyValuePair_2_HSTRING_HSTRING_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("60310303-49c5-52e6-abc6-a9b36eccc716"))
IKeyValuePair<HSTRING,HSTRING> : IKeyValuePair_impl<HSTRING,HSTRING> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IKeyValuePair`2<String, String>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IKeyValuePair<HSTRING,HSTRING> __FIKeyValuePair_2_HSTRING_HSTRING_t;
#define __FIKeyValuePair_2_HSTRING_HSTRING ABI::Windows::Foundation::Collections::__FIKeyValuePair_2_HSTRING_HSTRING_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIKeyValuePair_2_HSTRING_HSTRING ABI::Windows::Foundation::Collections::IKeyValuePair<HSTRING,HSTRING>
//#define __FIKeyValuePair_2_HSTRING_HSTRING_t ABI::Windows::Foundation::Collections::IKeyValuePair<HSTRING,HSTRING>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIKeyValuePair_2_HSTRING_HSTRING_USE */





#ifndef DEF___FIIterator_1___FIKeyValuePair_2_HSTRING_HSTRING_USE
#define DEF___FIIterator_1___FIKeyValuePair_2_HSTRING_HSTRING_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("05eb86f1-7140-5517-b88d-cbaebe57e6b1"))
IIterator<__FIKeyValuePair_2_HSTRING_HSTRING*> : IIterator_impl<__FIKeyValuePair_2_HSTRING_HSTRING*> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IIterator`1<Windows.Foundation.Collections.IKeyValuePair`2<String, String>>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IIterator<__FIKeyValuePair_2_HSTRING_HSTRING*> __FIIterator_1___FIKeyValuePair_2_HSTRING_HSTRING_t;
#define __FIIterator_1___FIKeyValuePair_2_HSTRING_HSTRING ABI::Windows::Foundation::Collections::__FIIterator_1___FIKeyValuePair_2_HSTRING_HSTRING_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIIterator_1___FIKeyValuePair_2_HSTRING_HSTRING ABI::Windows::Foundation::Collections::IIterator<ABI::Windows::Foundation::Collections::IKeyValuePair<HSTRING,HSTRING>*>
//#define __FIIterator_1___FIKeyValuePair_2_HSTRING_HSTRING_t ABI::Windows::Foundation::Collections::IIterator<ABI::Windows::Foundation::Collections::IKeyValuePair<HSTRING,HSTRING>*>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIIterator_1___FIKeyValuePair_2_HSTRING_HSTRING_USE */





#ifndef DEF___FIIterable_1___FIKeyValuePair_2_HSTRING_HSTRING_USE
#define DEF___FIIterable_1___FIKeyValuePair_2_HSTRING_HSTRING_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("e9bdaaf0-cbf6-5c72-be90-29cbf3a1319b"))
IIterable<__FIKeyValuePair_2_HSTRING_HSTRING*> : IIterable_impl<__FIKeyValuePair_2_HSTRING_HSTRING*> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IIterable`1<Windows.Foundation.Collections.IKeyValuePair`2<String, String>>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IIterable<__FIKeyValuePair_2_HSTRING_HSTRING*> __FIIterable_1___FIKeyValuePair_2_HSTRING_HSTRING_t;
#define __FIIterable_1___FIKeyValuePair_2_HSTRING_HSTRING ABI::Windows::Foundation::Collections::__FIIterable_1___FIKeyValuePair_2_HSTRING_HSTRING_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIIterable_1___FIKeyValuePair_2_HSTRING_HSTRING ABI::Windows::Foundation::Collections::IIterable<ABI::Windows::Foundation::Collections::IKeyValuePair<HSTRING,HSTRING>*>
//#define __FIIterable_1___FIKeyValuePair_2_HSTRING_HSTRING_t ABI::Windows::Foundation::Collections::IIterable<ABI::Windows::Foundation::Collections::IKeyValuePair<HSTRING,HSTRING>*>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIIterable_1___FIKeyValuePair_2_HSTRING_HSTRING_USE */




#ifndef DEF___FIMapView_2_HSTRING_HSTRING_USE
#define DEF___FIMapView_2_HSTRING_HSTRING_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("ac7f26f2-feb7-5b2a-8ac4-345bc62caede"))
IMapView<HSTRING,HSTRING> : IMapView_impl<HSTRING,HSTRING> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IMapView`2<String, String>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IMapView<HSTRING,HSTRING> __FIMapView_2_HSTRING_HSTRING_t;
#define __FIMapView_2_HSTRING_HSTRING ABI::Windows::Foundation::Collections::__FIMapView_2_HSTRING_HSTRING_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIMapView_2_HSTRING_HSTRING ABI::Windows::Foundation::Collections::IMapView<HSTRING,HSTRING>
//#define __FIMapView_2_HSTRING_HSTRING_t ABI::Windows::Foundation::Collections::IMapView<HSTRING,HSTRING>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIMapView_2_HSTRING_HSTRING_USE */




#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef DEF___FIIterator_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_USE
#define DEF___FIIterator_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("02996df2-340b-5f45-aa9e-0730e8934522"))
IIterator<ABI::Microsoft::AI::MachineLearning::ILearningModelFeatureDescriptor*> : IIterator_impl<ABI::Microsoft::AI::MachineLearning::ILearningModelFeatureDescriptor*> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IIterator`1<Microsoft.AI.MachineLearning.ILearningModelFeatureDescriptor>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IIterator<ABI::Microsoft::AI::MachineLearning::ILearningModelFeatureDescriptor*> __FIIterator_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_t;
#define __FIIterator_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor ABI::Windows::Foundation::Collections::__FIIterator_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIIterator_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor ABI::Windows::Foundation::Collections::IIterator<ABI::Microsoft::AI::MachineLearning::ILearningModelFeatureDescriptor*>
//#define __FIIterator_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_t ABI::Windows::Foundation::Collections::IIterator<ABI::Microsoft::AI::MachineLearning::ILearningModelFeatureDescriptor*>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIIterator_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_USE */


#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef DEF___FIIterable_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_USE
#define DEF___FIIterable_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("bcc32237-41be-5a28-a0aa-ec70bd4830a4"))
IIterable<ABI::Microsoft::AI::MachineLearning::ILearningModelFeatureDescriptor*> : IIterable_impl<ABI::Microsoft::AI::MachineLearning::ILearningModelFeatureDescriptor*> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IIterable`1<Microsoft.AI.MachineLearning.ILearningModelFeatureDescriptor>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IIterable<ABI::Microsoft::AI::MachineLearning::ILearningModelFeatureDescriptor*> __FIIterable_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_t;
#define __FIIterable_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor ABI::Windows::Foundation::Collections::__FIIterable_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIIterable_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor ABI::Windows::Foundation::Collections::IIterable<ABI::Microsoft::AI::MachineLearning::ILearningModelFeatureDescriptor*>
//#define __FIIterable_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_t ABI::Windows::Foundation::Collections::IIterable<ABI::Microsoft::AI::MachineLearning::ILearningModelFeatureDescriptor*>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIIterable_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_USE */


#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef DEF___FIVectorView_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_USE
#define DEF___FIVectorView_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("8ac44135-a995-5bcc-b6e4-930d94dc7fdb"))
IVectorView<ABI::Microsoft::AI::MachineLearning::ILearningModelFeatureDescriptor*> : IVectorView_impl<ABI::Microsoft::AI::MachineLearning::ILearningModelFeatureDescriptor*> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IVectorView`1<Microsoft.AI.MachineLearning.ILearningModelFeatureDescriptor>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IVectorView<ABI::Microsoft::AI::MachineLearning::ILearningModelFeatureDescriptor*> __FIVectorView_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_t;
#define __FIVectorView_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor ABI::Windows::Foundation::Collections::__FIVectorView_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIVectorView_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor ABI::Windows::Foundation::Collections::IVectorView<ABI::Microsoft::AI::MachineLearning::ILearningModelFeatureDescriptor*>
//#define __FIVectorView_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_t ABI::Windows::Foundation::Collections::IVectorView<ABI::Microsoft::AI::MachineLearning::ILearningModelFeatureDescriptor*>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIVectorView_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_USE */


#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                class LearningModel;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */


#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef DEF___FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModel_USE
#define DEF___FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModel_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation {
template <>
struct __declspec(uuid("cc2c54eb-6240-5fc3-a1ea-4871a72d9f28"))
IAsyncOperationCompletedHandler<ABI::Microsoft::AI::MachineLearning::LearningModel*> : IAsyncOperationCompletedHandler_impl<ABI::Windows::Foundation::Internal::AggregateType<ABI::Microsoft::AI::MachineLearning::LearningModel*, ABI::Microsoft::AI::MachineLearning::ILearningModel*>> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.AsyncOperationCompletedHandler`1<Microsoft.AI.MachineLearning.LearningModel>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IAsyncOperationCompletedHandler<ABI::Microsoft::AI::MachineLearning::LearningModel*> __FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModel_t;
#define __FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModel ABI::Windows::Foundation::__FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModel_t
/* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModel ABI::Windows::Foundation::IAsyncOperationCompletedHandler<ABI::Microsoft::AI::MachineLearning::ILearningModel*>
//#define __FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModel_t ABI::Windows::Foundation::IAsyncOperationCompletedHandler<ABI::Microsoft::AI::MachineLearning::ILearningModel*>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModel_USE */


#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef DEF___FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModel_USE
#define DEF___FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModel_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation {
template <>
struct __declspec(uuid("f2cbbb6c-457d-5132-a418-0349ca0012b2"))
IAsyncOperation<ABI::Microsoft::AI::MachineLearning::LearningModel*> : IAsyncOperation_impl<ABI::Windows::Foundation::Internal::AggregateType<ABI::Microsoft::AI::MachineLearning::LearningModel*, ABI::Microsoft::AI::MachineLearning::ILearningModel*>> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.IAsyncOperation`1<Microsoft.AI.MachineLearning.LearningModel>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IAsyncOperation<ABI::Microsoft::AI::MachineLearning::LearningModel*> __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModel_t;
#define __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModel ABI::Windows::Foundation::__FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModel_t
/* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModel ABI::Windows::Foundation::IAsyncOperation<ABI::Microsoft::AI::MachineLearning::ILearningModel*>
//#define __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModel_t ABI::Windows::Foundation::IAsyncOperation<ABI::Microsoft::AI::MachineLearning::ILearningModel*>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModel_USE */


#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000



#ifndef DEF___FIKeyValuePair_2_HSTRING_IInspectable_USE
#define DEF___FIKeyValuePair_2_HSTRING_IInspectable_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("09335560-6c6b-5a26-9348-97b781132b20"))
IKeyValuePair<HSTRING,IInspectable*> : IKeyValuePair_impl<HSTRING,IInspectable*> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IKeyValuePair`2<String, Object>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IKeyValuePair<HSTRING,IInspectable*> __FIKeyValuePair_2_HSTRING_IInspectable_t;
#define __FIKeyValuePair_2_HSTRING_IInspectable ABI::Windows::Foundation::Collections::__FIKeyValuePair_2_HSTRING_IInspectable_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIKeyValuePair_2_HSTRING_IInspectable ABI::Windows::Foundation::Collections::IKeyValuePair<HSTRING,IInspectable*>
//#define __FIKeyValuePair_2_HSTRING_IInspectable_t ABI::Windows::Foundation::Collections::IKeyValuePair<HSTRING,IInspectable*>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIKeyValuePair_2_HSTRING_IInspectable_USE */





#ifndef DEF___FIIterator_1___FIKeyValuePair_2_HSTRING_IInspectable_USE
#define DEF___FIIterator_1___FIKeyValuePair_2_HSTRING_IInspectable_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("5db5fa32-707c-5849-a06b-91c8eb9d10e8"))
IIterator<__FIKeyValuePair_2_HSTRING_IInspectable*> : IIterator_impl<__FIKeyValuePair_2_HSTRING_IInspectable*> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IIterator`1<Windows.Foundation.Collections.IKeyValuePair`2<String, Object>>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IIterator<__FIKeyValuePair_2_HSTRING_IInspectable*> __FIIterator_1___FIKeyValuePair_2_HSTRING_IInspectable_t;
#define __FIIterator_1___FIKeyValuePair_2_HSTRING_IInspectable ABI::Windows::Foundation::Collections::__FIIterator_1___FIKeyValuePair_2_HSTRING_IInspectable_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIIterator_1___FIKeyValuePair_2_HSTRING_IInspectable ABI::Windows::Foundation::Collections::IIterator<ABI::Windows::Foundation::Collections::IKeyValuePair<HSTRING,IInspectable*>*>
//#define __FIIterator_1___FIKeyValuePair_2_HSTRING_IInspectable_t ABI::Windows::Foundation::Collections::IIterator<ABI::Windows::Foundation::Collections::IKeyValuePair<HSTRING,IInspectable*>*>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIIterator_1___FIKeyValuePair_2_HSTRING_IInspectable_USE */





#ifndef DEF___FIIterable_1___FIKeyValuePair_2_HSTRING_IInspectable_USE
#define DEF___FIIterable_1___FIKeyValuePair_2_HSTRING_IInspectable_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("fe2f3d47-5d47-5499-8374-430c7cda0204"))
IIterable<__FIKeyValuePair_2_HSTRING_IInspectable*> : IIterable_impl<__FIKeyValuePair_2_HSTRING_IInspectable*> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IIterable`1<Windows.Foundation.Collections.IKeyValuePair`2<String, Object>>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IIterable<__FIKeyValuePair_2_HSTRING_IInspectable*> __FIIterable_1___FIKeyValuePair_2_HSTRING_IInspectable_t;
#define __FIIterable_1___FIKeyValuePair_2_HSTRING_IInspectable ABI::Windows::Foundation::Collections::__FIIterable_1___FIKeyValuePair_2_HSTRING_IInspectable_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIIterable_1___FIKeyValuePair_2_HSTRING_IInspectable ABI::Windows::Foundation::Collections::IIterable<ABI::Windows::Foundation::Collections::IKeyValuePair<HSTRING,IInspectable*>*>
//#define __FIIterable_1___FIKeyValuePair_2_HSTRING_IInspectable_t ABI::Windows::Foundation::Collections::IIterable<ABI::Windows::Foundation::Collections::IKeyValuePair<HSTRING,IInspectable*>*>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIIterable_1___FIKeyValuePair_2_HSTRING_IInspectable_USE */





#ifndef DEF___FIMapView_2_HSTRING_IInspectable_USE
#define DEF___FIMapView_2_HSTRING_IInspectable_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("bb78502a-f79d-54fa-92c9-90c5039fdf7e"))
IMapView<HSTRING,IInspectable*> : IMapView_impl<HSTRING,IInspectable*> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IMapView`2<String, Object>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IMapView<HSTRING,IInspectable*> __FIMapView_2_HSTRING_IInspectable_t;
#define __FIMapView_2_HSTRING_IInspectable ABI::Windows::Foundation::Collections::__FIMapView_2_HSTRING_IInspectable_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIMapView_2_HSTRING_IInspectable ABI::Windows::Foundation::Collections::IMapView<HSTRING,IInspectable*>
//#define __FIMapView_2_HSTRING_IInspectable_t ABI::Windows::Foundation::Collections::IMapView<HSTRING,IInspectable*>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIMapView_2_HSTRING_IInspectable_USE */



namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                class LearningModelEvaluationResult;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */


#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef DEF___FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult_USE
#define DEF___FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation {
template <>
struct __declspec(uuid("37c7bcdc-3d7e-5cbb-b3f6-f00c60a3f81e"))
IAsyncOperationCompletedHandler<ABI::Microsoft::AI::MachineLearning::LearningModelEvaluationResult*> : IAsyncOperationCompletedHandler_impl<ABI::Windows::Foundation::Internal::AggregateType<ABI::Microsoft::AI::MachineLearning::LearningModelEvaluationResult*, ABI::Microsoft::AI::MachineLearning::ILearningModelEvaluationResult*>> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.AsyncOperationCompletedHandler`1<Microsoft.AI.MachineLearning.LearningModelEvaluationResult>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IAsyncOperationCompletedHandler<ABI::Microsoft::AI::MachineLearning::LearningModelEvaluationResult*> __FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult_t;
#define __FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult ABI::Windows::Foundation::__FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult_t
/* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult ABI::Windows::Foundation::IAsyncOperationCompletedHandler<ABI::Microsoft::AI::MachineLearning::ILearningModelEvaluationResult*>
//#define __FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult_t ABI::Windows::Foundation::IAsyncOperationCompletedHandler<ABI::Microsoft::AI::MachineLearning::ILearningModelEvaluationResult*>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult_USE */


#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef DEF___FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult_USE
#define DEF___FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation {
template <>
struct __declspec(uuid("67ee8ad3-251c-5e18-8be4-6dbfb9b57ab3"))
IAsyncOperation<ABI::Microsoft::AI::MachineLearning::LearningModelEvaluationResult*> : IAsyncOperation_impl<ABI::Windows::Foundation::Internal::AggregateType<ABI::Microsoft::AI::MachineLearning::LearningModelEvaluationResult*, ABI::Microsoft::AI::MachineLearning::ILearningModelEvaluationResult*>> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.IAsyncOperation`1<Microsoft.AI.MachineLearning.LearningModelEvaluationResult>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IAsyncOperation<ABI::Microsoft::AI::MachineLearning::LearningModelEvaluationResult*> __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult_t;
#define __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult ABI::Windows::Foundation::__FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult_t
/* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult ABI::Windows::Foundation::IAsyncOperation<ABI::Microsoft::AI::MachineLearning::ILearningModelEvaluationResult*>
//#define __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult_t ABI::Windows::Foundation::IAsyncOperation<ABI::Microsoft::AI::MachineLearning::ILearningModelEvaluationResult*>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult_USE */


#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000



#ifndef DEF___FIMap_2_HSTRING_IInspectable_USE
#define DEF___FIMap_2_HSTRING_IInspectable_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("1b0d3570-0877-5ec2-8a2c-3b9539506aca"))
IMap<HSTRING,IInspectable*> : IMap_impl<HSTRING,IInspectable*> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IMap`2<String, Object>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IMap<HSTRING,IInspectable*> __FIMap_2_HSTRING_IInspectable_t;
#define __FIMap_2_HSTRING_IInspectable ABI::Windows::Foundation::Collections::__FIMap_2_HSTRING_IInspectable_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIMap_2_HSTRING_IInspectable ABI::Windows::Foundation::Collections::IMap<HSTRING,IInspectable*>
//#define __FIMap_2_HSTRING_IInspectable_t ABI::Windows::Foundation::Collections::IMap<HSTRING,IInspectable*>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIMap_2_HSTRING_IInspectable_USE */




#ifndef DEF___FIIterator_1___z__zint64_USE
#define DEF___FIIterator_1___z__zint64_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("fb98034c-86b7-581f-8cd9-5ad0692201a9"))
IIterator<__int64> : IIterator_impl<__int64> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IIterator`1<Int64>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IIterator<__int64> __FIIterator_1___z__zint64_t;
#define __FIIterator_1___z__zint64 ABI::Windows::Foundation::Collections::__FIIterator_1___z__zint64_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIIterator_1___z__zint64 ABI::Windows::Foundation::Collections::IIterator<INT64>
//#define __FIIterator_1___z__zint64_t ABI::Windows::Foundation::Collections::IIterator<INT64>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIIterator_1___z__zint64_USE */




#ifndef DEF___FIIterable_1___z__zint64_USE
#define DEF___FIIterable_1___z__zint64_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("7784427e-f9cc-518d-964b-e50d5ce727f1"))
IIterable<__int64> : IIterable_impl<__int64> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IIterable`1<Int64>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IIterable<__int64> __FIIterable_1___z__zint64_t;
#define __FIIterable_1___z__zint64 ABI::Windows::Foundation::Collections::__FIIterable_1___z__zint64_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIIterable_1___z__zint64 ABI::Windows::Foundation::Collections::IIterable<INT64>
//#define __FIIterable_1___z__zint64_t ABI::Windows::Foundation::Collections::IIterable<INT64>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIIterable_1___z__zint64_USE */




#ifndef DEF___FIVectorView_1___z__zint64_USE
#define DEF___FIVectorView_1___z__zint64_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("8221aa0e-d1d2-5b22-a918-05672812d12f"))
IVectorView<__int64> : IVectorView_impl<__int64> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IVectorView`1<Int64>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IVectorView<__int64> __FIVectorView_1___z__zint64_t;
#define __FIVectorView_1___z__zint64 ABI::Windows::Foundation::Collections::__FIVectorView_1___z__zint64_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIVectorView_1___z__zint64 ABI::Windows::Foundation::Collections::IVectorView<INT64>
//#define __FIVectorView_1___z__zint64_t ABI::Windows::Foundation::Collections::IVectorView<INT64>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIVectorView_1___z__zint64_USE */




#ifndef DEF___FIIterator_1_float_USE
#define DEF___FIIterator_1_float_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("42614e61-b0aa-5e72-9354-2771db20b7a8"))
IIterator<float> : IIterator_impl<float> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IIterator`1<Single>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IIterator<float> __FIIterator_1_float_t;
#define __FIIterator_1_float ABI::Windows::Foundation::Collections::__FIIterator_1_float_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIIterator_1_float ABI::Windows::Foundation::Collections::IIterator<FLOAT>
//#define __FIIterator_1_float_t ABI::Windows::Foundation::Collections::IIterator<FLOAT>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIIterator_1_float_USE */




#ifndef DEF___FIIterable_1_float_USE
#define DEF___FIIterable_1_float_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("b01bee51-063a-5fda-bd72-d76637bb8cb8"))
IIterable<float> : IIterable_impl<float> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IIterable`1<Single>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IIterable<float> __FIIterable_1_float_t;
#define __FIIterable_1_float ABI::Windows::Foundation::Collections::__FIIterable_1_float_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIIterable_1_float ABI::Windows::Foundation::Collections::IIterable<FLOAT>
//#define __FIIterable_1_float_t ABI::Windows::Foundation::Collections::IIterable<FLOAT>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIIterable_1_float_USE */




#ifndef DEF___FIVectorView_1_float_USE
#define DEF___FIVectorView_1_float_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("7bca64fd-150c-5d50-b56b-9f4f474c5930"))
IVectorView<float> : IVectorView_impl<float> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IVectorView`1<Single>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IVectorView<float> __FIVectorView_1_float_t;
#define __FIVectorView_1_float ABI::Windows::Foundation::Collections::__FIVectorView_1_float_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIVectorView_1_float ABI::Windows::Foundation::Collections::IVectorView<FLOAT>
//#define __FIVectorView_1_float_t ABI::Windows::Foundation::Collections::IVectorView<FLOAT>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIVectorView_1_float_USE */




#ifndef DEF___FIIterator_1_byte_USE
#define DEF___FIIterator_1_byte_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("40556131-a2a1-5fab-aaee-5f35268ca26b"))
IIterator<::byte> : IIterator_impl<::byte> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IIterator`1<UInt8>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IIterator<::byte> __FIIterator_1_byte_t;
#define __FIIterator_1_byte ABI::Windows::Foundation::Collections::__FIIterator_1_byte_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIIterator_1_byte ABI::Windows::Foundation::Collections::IIterator<BYTE>
//#define __FIIterator_1_byte_t ABI::Windows::Foundation::Collections::IIterator<BYTE>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIIterator_1_byte_USE */




#ifndef DEF___FIIterable_1_byte_USE
#define DEF___FIIterable_1_byte_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("88318266-f3fd-50fc-8f08-b823a41b60c1"))
IIterable<::byte> : IIterable_impl<::byte> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IIterable`1<UInt8>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IIterable<::byte> __FIIterable_1_byte_t;
#define __FIIterable_1_byte ABI::Windows::Foundation::Collections::__FIIterable_1_byte_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIIterable_1_byte ABI::Windows::Foundation::Collections::IIterable<BYTE>
//#define __FIIterable_1_byte_t ABI::Windows::Foundation::Collections::IIterable<BYTE>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIIterable_1_byte_USE */




#ifndef DEF___FIVectorView_1_byte_USE
#define DEF___FIVectorView_1_byte_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("6d05fb29-7885-544e-9382-a1ad391a3fa4"))
IVectorView<::byte> : IVectorView_impl<::byte> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IVectorView`1<UInt8>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IVectorView<::byte> __FIVectorView_1_byte_t;
#define __FIVectorView_1_byte ABI::Windows::Foundation::Collections::__FIVectorView_1_byte_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIVectorView_1_byte ABI::Windows::Foundation::Collections::IVectorView<BYTE>
//#define __FIVectorView_1_byte_t ABI::Windows::Foundation::Collections::IVectorView<BYTE>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIVectorView_1_byte_USE */




#ifndef DEF___FIIterator_1_UINT16_USE
#define DEF___FIIterator_1_UINT16_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("5738fc25-402b-5fc1-b1e4-0aa24ef652f1"))
IIterator<UINT16> : IIterator_impl<UINT16> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IIterator`1<UInt16>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IIterator<UINT16> __FIIterator_1_UINT16_t;
#define __FIIterator_1_UINT16 ABI::Windows::Foundation::Collections::__FIIterator_1_UINT16_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIIterator_1_UINT16 ABI::Windows::Foundation::Collections::IIterator<UINT16>
//#define __FIIterator_1_UINT16_t ABI::Windows::Foundation::Collections::IIterator<UINT16>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIIterator_1_UINT16_USE */




#ifndef DEF___FIIterable_1_UINT16_USE
#define DEF___FIIterable_1_UINT16_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("ecfa9a6f-fa2e-5345-b297-efb4e8c6be87"))
IIterable<UINT16> : IIterable_impl<UINT16> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IIterable`1<UInt16>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IIterable<UINT16> __FIIterable_1_UINT16_t;
#define __FIIterable_1_UINT16 ABI::Windows::Foundation::Collections::__FIIterable_1_UINT16_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIIterable_1_UINT16 ABI::Windows::Foundation::Collections::IIterable<UINT16>
//#define __FIIterable_1_UINT16_t ABI::Windows::Foundation::Collections::IIterable<UINT16>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIIterable_1_UINT16_USE */




#ifndef DEF___FIVectorView_1_UINT16_USE
#define DEF___FIVectorView_1_UINT16_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("9d0d0d9f-6a82-55a3-98c5-228499df38f9"))
IVectorView<UINT16> : IVectorView_impl<UINT16> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IVectorView`1<UInt16>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IVectorView<UINT16> __FIVectorView_1_UINT16_t;
#define __FIVectorView_1_UINT16 ABI::Windows::Foundation::Collections::__FIVectorView_1_UINT16_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIVectorView_1_UINT16 ABI::Windows::Foundation::Collections::IVectorView<UINT16>
//#define __FIVectorView_1_UINT16_t ABI::Windows::Foundation::Collections::IVectorView<UINT16>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIVectorView_1_UINT16_USE */




#ifndef DEF___FIIterator_1_short_USE
#define DEF___FIIterator_1_short_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("5409069f-e7c1-5732-bb69-e5736f03f9a9"))
IIterator<short> : IIterator_impl<short> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IIterator`1<Int16>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IIterator<short> __FIIterator_1_short_t;
#define __FIIterator_1_short ABI::Windows::Foundation::Collections::__FIIterator_1_short_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIIterator_1_short ABI::Windows::Foundation::Collections::IIterator<INT16>
//#define __FIIterator_1_short_t ABI::Windows::Foundation::Collections::IIterator<INT16>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIIterator_1_short_USE */




#ifndef DEF___FIIterable_1_short_USE
#define DEF___FIIterable_1_short_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("72ff2923-4b4e-53bb-8feb-41ec5f2bb734"))
IIterable<short> : IIterable_impl<short> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IIterable`1<Int16>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IIterable<short> __FIIterable_1_short_t;
#define __FIIterable_1_short ABI::Windows::Foundation::Collections::__FIIterable_1_short_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIIterable_1_short ABI::Windows::Foundation::Collections::IIterable<INT16>
//#define __FIIterable_1_short_t ABI::Windows::Foundation::Collections::IIterable<INT16>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIIterable_1_short_USE */




#ifndef DEF___FIVectorView_1_short_USE
#define DEF___FIVectorView_1_short_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("e53056ad-8a0e-5c41-a62d-c92e3ac2de58"))
IVectorView<short> : IVectorView_impl<short> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IVectorView`1<Int16>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IVectorView<short> __FIVectorView_1_short_t;
#define __FIVectorView_1_short ABI::Windows::Foundation::Collections::__FIVectorView_1_short_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIVectorView_1_short ABI::Windows::Foundation::Collections::IVectorView<INT16>
//#define __FIVectorView_1_short_t ABI::Windows::Foundation::Collections::IVectorView<INT16>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIVectorView_1_short_USE */




#ifndef DEF___FIIterator_1_UINT32_USE
#define DEF___FIIterator_1_UINT32_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("f06a2739-9443-5ef0-b284-dc5aff3e7d10"))
IIterator<UINT32> : IIterator_impl<UINT32> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IIterator`1<UInt32>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IIterator<UINT32> __FIIterator_1_UINT32_t;
#define __FIIterator_1_UINT32 ABI::Windows::Foundation::Collections::__FIIterator_1_UINT32_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIIterator_1_UINT32 ABI::Windows::Foundation::Collections::IIterator<UINT32>
//#define __FIIterator_1_UINT32_t ABI::Windows::Foundation::Collections::IIterator<UINT32>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIIterator_1_UINT32_USE */




#ifndef DEF___FIIterable_1_UINT32_USE
#define DEF___FIIterable_1_UINT32_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("421d4b91-b13b-5f37-ae54-b5249bd80539"))
IIterable<UINT32> : IIterable_impl<UINT32> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IIterable`1<UInt32>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IIterable<UINT32> __FIIterable_1_UINT32_t;
#define __FIIterable_1_UINT32 ABI::Windows::Foundation::Collections::__FIIterable_1_UINT32_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIIterable_1_UINT32 ABI::Windows::Foundation::Collections::IIterable<UINT32>
//#define __FIIterable_1_UINT32_t ABI::Windows::Foundation::Collections::IIterable<UINT32>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIIterable_1_UINT32_USE */




#ifndef DEF___FIVectorView_1_UINT32_USE
#define DEF___FIVectorView_1_UINT32_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("e5ce1a07-8d33-5007-ba64-7d2508ccf85c"))
IVectorView<UINT32> : IVectorView_impl<UINT32> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IVectorView`1<UInt32>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IVectorView<UINT32> __FIVectorView_1_UINT32_t;
#define __FIVectorView_1_UINT32 ABI::Windows::Foundation::Collections::__FIVectorView_1_UINT32_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIVectorView_1_UINT32 ABI::Windows::Foundation::Collections::IVectorView<UINT32>
//#define __FIVectorView_1_UINT32_t ABI::Windows::Foundation::Collections::IVectorView<UINT32>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIVectorView_1_UINT32_USE */




#ifndef DEF___FIIterator_1_int_USE
#define DEF___FIIterator_1_int_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("bfea7f78-50c2-5f1d-a6ea-9e978d2699ff"))
IIterator<int> : IIterator_impl<int> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IIterator`1<Int32>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IIterator<int> __FIIterator_1_int_t;
#define __FIIterator_1_int ABI::Windows::Foundation::Collections::__FIIterator_1_int_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIIterator_1_int ABI::Windows::Foundation::Collections::IIterator<INT32>
//#define __FIIterator_1_int_t ABI::Windows::Foundation::Collections::IIterator<INT32>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIIterator_1_int_USE */




#ifndef DEF___FIIterable_1_int_USE
#define DEF___FIIterable_1_int_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("81a643fb-f51c-5565-83c4-f96425777b66"))
IIterable<int> : IIterable_impl<int> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IIterable`1<Int32>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IIterable<int> __FIIterable_1_int_t;
#define __FIIterable_1_int ABI::Windows::Foundation::Collections::__FIIterable_1_int_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIIterable_1_int ABI::Windows::Foundation::Collections::IIterable<INT32>
//#define __FIIterable_1_int_t ABI::Windows::Foundation::Collections::IIterable<INT32>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIIterable_1_int_USE */




#ifndef DEF___FIVectorView_1_int_USE
#define DEF___FIVectorView_1_int_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("8d720cdf-3934-5d3f-9a55-40e8063b086a"))
IVectorView<int> : IVectorView_impl<int> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IVectorView`1<Int32>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IVectorView<int> __FIVectorView_1_int_t;
#define __FIVectorView_1_int ABI::Windows::Foundation::Collections::__FIVectorView_1_int_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIVectorView_1_int ABI::Windows::Foundation::Collections::IVectorView<INT32>
//#define __FIVectorView_1_int_t ABI::Windows::Foundation::Collections::IVectorView<INT32>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIVectorView_1_int_USE */




#ifndef DEF___FIIterator_1_UINT64_USE
#define DEF___FIIterator_1_UINT64_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("c473ed96-76e3-5ff2-9435-47feebfe9539"))
IIterator<UINT64> : IIterator_impl<UINT64> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IIterator`1<UInt64>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IIterator<UINT64> __FIIterator_1_UINT64_t;
#define __FIIterator_1_UINT64 ABI::Windows::Foundation::Collections::__FIIterator_1_UINT64_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIIterator_1_UINT64 ABI::Windows::Foundation::Collections::IIterator<UINT64>
//#define __FIIterator_1_UINT64_t ABI::Windows::Foundation::Collections::IIterator<UINT64>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIIterator_1_UINT64_USE */




#ifndef DEF___FIIterable_1_UINT64_USE
#define DEF___FIIterable_1_UINT64_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("4b3a3229-7995-5f3c-b248-6c1f7e664f01"))
IIterable<UINT64> : IIterable_impl<UINT64> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IIterable`1<UInt64>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IIterable<UINT64> __FIIterable_1_UINT64_t;
#define __FIIterable_1_UINT64 ABI::Windows::Foundation::Collections::__FIIterable_1_UINT64_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIIterable_1_UINT64 ABI::Windows::Foundation::Collections::IIterable<UINT64>
//#define __FIIterable_1_UINT64_t ABI::Windows::Foundation::Collections::IIterable<UINT64>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIIterable_1_UINT64_USE */




#ifndef DEF___FIVectorView_1_UINT64_USE
#define DEF___FIVectorView_1_UINT64_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("23d156c7-7ef9-5096-aaba-1e6c9ab5ceb4"))
IVectorView<UINT64> : IVectorView_impl<UINT64> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IVectorView`1<UInt64>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IVectorView<UINT64> __FIVectorView_1_UINT64_t;
#define __FIVectorView_1_UINT64 ABI::Windows::Foundation::Collections::__FIVectorView_1_UINT64_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIVectorView_1_UINT64 ABI::Windows::Foundation::Collections::IVectorView<UINT64>
//#define __FIVectorView_1_UINT64_t ABI::Windows::Foundation::Collections::IVectorView<UINT64>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIVectorView_1_UINT64_USE */




#ifndef DEF___FIIterator_1_boolean_USE
#define DEF___FIIterator_1_boolean_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("740a0296-a535-572a-bf0b-17c18ff71fe6"))
IIterator<bool> : IIterator_impl<ABI::Windows::Foundation::Internal::AggregateType<bool, boolean>> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IIterator`1<Boolean>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IIterator<bool> __FIIterator_1_boolean_t;
#define __FIIterator_1_boolean ABI::Windows::Foundation::Collections::__FIIterator_1_boolean_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIIterator_1_boolean ABI::Windows::Foundation::Collections::IIterator<boolean>
//#define __FIIterator_1_boolean_t ABI::Windows::Foundation::Collections::IIterator<boolean>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIIterator_1_boolean_USE */




#ifndef DEF___FIIterable_1_boolean_USE
#define DEF___FIIterable_1_boolean_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("30160817-1d7d-54e9-99db-d7636266a476"))
IIterable<bool> : IIterable_impl<ABI::Windows::Foundation::Internal::AggregateType<bool, boolean>> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IIterable`1<Boolean>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IIterable<bool> __FIIterable_1_boolean_t;
#define __FIIterable_1_boolean ABI::Windows::Foundation::Collections::__FIIterable_1_boolean_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIIterable_1_boolean ABI::Windows::Foundation::Collections::IIterable<boolean>
//#define __FIIterable_1_boolean_t ABI::Windows::Foundation::Collections::IIterable<boolean>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIIterable_1_boolean_USE */




#ifndef DEF___FIVectorView_1_boolean_USE
#define DEF___FIVectorView_1_boolean_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("243a09cb-6f40-56af-a442-fe81431fbef5"))
IVectorView<bool> : IVectorView_impl<ABI::Windows::Foundation::Internal::AggregateType<bool, boolean>> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IVectorView`1<Boolean>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IVectorView<bool> __FIVectorView_1_boolean_t;
#define __FIVectorView_1_boolean ABI::Windows::Foundation::Collections::__FIVectorView_1_boolean_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIVectorView_1_boolean ABI::Windows::Foundation::Collections::IVectorView<boolean>
//#define __FIVectorView_1_boolean_t ABI::Windows::Foundation::Collections::IVectorView<boolean>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIVectorView_1_boolean_USE */




#ifndef DEF___FIIterator_1_double_USE
#define DEF___FIIterator_1_double_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("638a2cf4-f474-5318-9055-141cb909ac4b"))
IIterator<double> : IIterator_impl<double> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IIterator`1<Double>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IIterator<double> __FIIterator_1_double_t;
#define __FIIterator_1_double ABI::Windows::Foundation::Collections::__FIIterator_1_double_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIIterator_1_double ABI::Windows::Foundation::Collections::IIterator<DOUBLE>
//#define __FIIterator_1_double_t ABI::Windows::Foundation::Collections::IIterator<DOUBLE>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIIterator_1_double_USE */




#ifndef DEF___FIIterable_1_double_USE
#define DEF___FIIterable_1_double_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("c738964e-9c64-5bce-b5ce-61e9a282ec4a"))
IIterable<double> : IIterable_impl<double> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IIterable`1<Double>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IIterable<double> __FIIterable_1_double_t;
#define __FIIterable_1_double ABI::Windows::Foundation::Collections::__FIIterable_1_double_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIIterable_1_double ABI::Windows::Foundation::Collections::IIterable<DOUBLE>
//#define __FIIterable_1_double_t ABI::Windows::Foundation::Collections::IIterable<DOUBLE>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIIterable_1_double_USE */




#ifndef DEF___FIVectorView_1_double_USE
#define DEF___FIVectorView_1_double_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("af7586a8-6b21-5f61-bff1-1b682293ad96"))
IVectorView<double> : IVectorView_impl<double> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IVectorView`1<Double>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IVectorView<double> __FIVectorView_1_double_t;
#define __FIVectorView_1_double ABI::Windows::Foundation::Collections::__FIVectorView_1_double_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIVectorView_1_double ABI::Windows::Foundation::Collections::IVectorView<DOUBLE>
//#define __FIVectorView_1_double_t ABI::Windows::Foundation::Collections::IVectorView<DOUBLE>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIVectorView_1_double_USE */




#ifndef DEF___FIIterator_1_HSTRING_USE
#define DEF___FIIterator_1_HSTRING_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("8c304ebb-6615-50a4-8829-879ecd443236"))
IIterator<HSTRING> : IIterator_impl<HSTRING> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IIterator`1<String>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IIterator<HSTRING> __FIIterator_1_HSTRING_t;
#define __FIIterator_1_HSTRING ABI::Windows::Foundation::Collections::__FIIterator_1_HSTRING_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIIterator_1_HSTRING ABI::Windows::Foundation::Collections::IIterator<HSTRING>
//#define __FIIterator_1_HSTRING_t ABI::Windows::Foundation::Collections::IIterator<HSTRING>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIIterator_1_HSTRING_USE */




#ifndef DEF___FIIterable_1_HSTRING_USE
#define DEF___FIIterable_1_HSTRING_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("e2fcc7c1-3bfc-5a0b-b2b0-72e769d1cb7e"))
IIterable<HSTRING> : IIterable_impl<HSTRING> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IIterable`1<String>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IIterable<HSTRING> __FIIterable_1_HSTRING_t;
#define __FIIterable_1_HSTRING ABI::Windows::Foundation::Collections::__FIIterable_1_HSTRING_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIIterable_1_HSTRING ABI::Windows::Foundation::Collections::IIterable<HSTRING>
//#define __FIIterable_1_HSTRING_t ABI::Windows::Foundation::Collections::IIterable<HSTRING>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIIterable_1_HSTRING_USE */




#ifndef DEF___FIVectorView_1_HSTRING_USE
#define DEF___FIVectorView_1_HSTRING_USE
#if !defined(RO_NO_TEMPLATE_NAME)
namespace ABI { namespace Windows { namespace Foundation { namespace Collections {
template <>
struct __declspec(uuid("2f13c006-a03a-5f69-b090-75a43e33423e"))
IVectorView<HSTRING> : IVectorView_impl<HSTRING> 
{
    static const wchar_t* z_get_rc_name_impl() 
    {
        return L"Windows.Foundation.Collections.IVectorView`1<String>"; 
    }
};
// Define a typedef for the parameterized interface specialization's mangled name.
// This allows code which uses the mangled name for the parameterized interface to access the
// correct parameterized interface specialization.
typedef IVectorView<HSTRING> __FIVectorView_1_HSTRING_t;
#define __FIVectorView_1_HSTRING ABI::Windows::Foundation::Collections::__FIVectorView_1_HSTRING_t
/* Collections */ } /* Foundation */ } /* Windows */ } /* ABI */ } 

////  Define an alias for the C version of the interface for compatibility purposes.
//#define __FIVectorView_1_HSTRING ABI::Windows::Foundation::Collections::IVectorView<HSTRING>
//#define __FIVectorView_1_HSTRING_t ABI::Windows::Foundation::Collections::IVectorView<HSTRING>
#endif // !defined(RO_NO_TEMPLATE_NAME)
#endif /* DEF___FIVectorView_1_HSTRING_USE */






#pragma warning (push)
#pragma warning (disable:4668) 
#pragma warning (disable:4001) 
#pragma once 
#pragma warning (pop)

namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                class LearningModelBinding;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */



/*
 *
 * Struct Microsoft.AI.MachineLearning.LearningModelFeatureKind
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 */

#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [v1_enum, contract] */
                enum LearningModelFeatureKind : int
                {
                    LearningModelFeatureKind_Tensor = 0,
                    LearningModelFeatureKind_Sequence,
                    LearningModelFeatureKind_Map,
                    LearningModelFeatureKind_Image,
                };
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ILearningModelFeatureDescriptor
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ILearningModelFeatureDescriptor:HRESULT get_Name(String*);HRESULT get_Description(String*);HRESULT get_Kind(Microsoft.AI.MachineLearning.LearningModelFeatureKind*);HRESULT get_IsRequired(Boolean*);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ILearningModelFeatureDescriptor[] = L"Microsoft.AI.MachineLearning.ILearningModelFeatureDescriptor";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("d21da0b6-b8b9-534b-97da-b3f0e0ffbac2"), object, contract] */
                MIDL_INTERFACE("d21da0b6-b8b9-534b-97da-b3f0e0ffbac2")
                ILearningModelFeatureDescriptor : public IInspectable
                {
                public:
                    /* [propget] */virtual HRESULT STDMETHODCALLTYPE get_Name(
                        /* [out, retval] */HSTRING * value
                        ) = 0;
                    /* [propget] */virtual HRESULT STDMETHODCALLTYPE get_Description(
                        /* [out, retval] */HSTRING * value
                        ) = 0;
                    /* [propget] */virtual HRESULT STDMETHODCALLTYPE get_Kind(
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::LearningModelFeatureKind * value
                        ) = 0;
                    /* [propget] */virtual HRESULT STDMETHODCALLTYPE get_IsRequired(
                        /* [out, retval] */::boolean * value
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ILearningModelFeatureDescriptor=_uuidof(ILearningModelFeatureDescriptor);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ILearningModelOperatorProvider
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelOperatorProvider_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelOperatorProvider_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ILearningModelOperatorProvider[] = L"Microsoft.AI.MachineLearning.ILearningModelOperatorProvider";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [object, contract, uuid("ae066239-6b19-4509-be3e-502ba40203b3")] */
                MIDL_INTERFACE("ae066239-6b19-4509-be3e-502ba40203b3")
                ILearningModelOperatorProvider : public IInspectable
                {
                public:
                    
                };

                extern MIDL_CONST_ID IID & IID_ILearningModelOperatorProvider=_uuidof(ILearningModelOperatorProvider);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelOperatorProvider;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelOperatorProvider_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000



/*
 *
 * Interface Microsoft.AI.MachineLearning.ILearningModel
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.LearningModel
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ILearningModel:HRESULT get_Author(String*);HRESULT get_Name(String*);HRESULT get_Domain(String*);HRESULT get_Description(String*);HRESULT get_Version(Int64*);HRESULT get_Metadata(Windows.Foundation.Collections.IMapView`2<String, String>**);HRESULT get_InputFeatures(Windows.Foundation.Collections.IVectorView`1<Microsoft.AI.MachineLearning.ILearningModelFeatureDescriptor>**);HRESULT get_OutputFeatures(Windows.Foundation.Collections.IVectorView`1<Microsoft.AI.MachineLearning.ILearningModelFeatureDescriptor>**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ILearningModel[] = L"Microsoft.AI.MachineLearning.ILearningModel";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("3e81e9c7-6af9-5bc0-941a-0b1857abb2d8"), contract, object, exclusiveto] */
                MIDL_INTERFACE("3e81e9c7-6af9-5bc0-941a-0b1857abb2d8")
                ILearningModel : public IInspectable
                {
                public:
                    /* [propget] */virtual HRESULT STDMETHODCALLTYPE get_Author(
                        /* [out, retval] */HSTRING * value
                        ) = 0;
                    /* [propget] */virtual HRESULT STDMETHODCALLTYPE get_Name(
                        /* [out, retval] */HSTRING * value
                        ) = 0;
                    /* [propget] */virtual HRESULT STDMETHODCALLTYPE get_Domain(
                        /* [out, retval] */HSTRING * value
                        ) = 0;
                    /* [propget] */virtual HRESULT STDMETHODCALLTYPE get_Description(
                        /* [out, retval] */HSTRING * value
                        ) = 0;
                    /* [propget] */virtual HRESULT STDMETHODCALLTYPE get_Version(
                        /* [out, retval] */__int64 * value
                        ) = 0;
                    /* [propget] */virtual HRESULT STDMETHODCALLTYPE get_Metadata(
                        /* [out, retval] */__FIMapView_2_HSTRING_HSTRING * * value
                        ) = 0;
                    /* [propget] */virtual HRESULT STDMETHODCALLTYPE get_InputFeatures(
                        /* [out, retval] */__FIVectorView_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor * * value
                        ) = 0;
                    /* [propget] */virtual HRESULT STDMETHODCALLTYPE get_OutputFeatures(
                        /* [out, retval] */__FIVectorView_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor * * value
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ILearningModel=_uuidof(ILearningModel);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ILearningModelStatics
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.LearningModel
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ILearningModelStatics:HRESULT LoadFromStorageFileAsync(Windows.Storage.IStorageFile*,Windows.Foundation.IAsyncOperation`1<Microsoft.AI.MachineLearning.LearningModel>**);HRESULT LoadFromStreamAsync(Windows.Storage.Streams.IRandomAccessStreamReference*,Windows.Foundation.IAsyncOperation`1<Microsoft.AI.MachineLearning.LearningModel>**);HRESULT LoadFromFilePath(String,Microsoft.AI.MachineLearning.LearningModel**);HRESULT LoadFromStream(Windows.Storage.Streams.IRandomAccessStreamReference*,Microsoft.AI.MachineLearning.LearningModel**);HRESULT LoadFromStorageFileWithOperatorProviderAsync(Windows.Storage.IStorageFile*,Microsoft.AI.MachineLearning.ILearningModelOperatorProvider*,Windows.Foundation.IAsyncOperation`1<Microsoft.AI.MachineLearning.LearningModel>**);HRESULT LoadFromStreamWithOperatorProviderAsync(Windows.Storage.Streams.IRandomAccessStreamReference*,Microsoft.AI.MachineLearning.ILearningModelOperatorProvider*,Windows.Foundation.IAsyncOperation`1<Microsoft.AI.MachineLearning.LearningModel>**);HRESULT LoadFromFilePathWithOperatorProvider(String,Microsoft.AI.MachineLearning.ILearningModelOperatorProvider*,Microsoft.AI.MachineLearning.LearningModel**);HRESULT LoadFromStreamWithOperatorProvider(Windows.Storage.Streams.IRandomAccessStreamReference*,Microsoft.AI.MachineLearning.ILearningModelOperatorProvider*,Microsoft.AI.MachineLearning.LearningModel**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ILearningModelStatics[] = L"Microsoft.AI.MachineLearning.ILearningModelStatics";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("ae329fdc-7a71-5b31-bbd9-bf0af125a6c3"), contract, object, exclusiveto] */
                MIDL_INTERFACE("ae329fdc-7a71-5b31-bbd9-bf0af125a6c3")
                ILearningModelStatics : public IInspectable
                {
                public:
                    /* [overload, remote_async] */virtual HRESULT STDMETHODCALLTYPE LoadFromStorageFileAsync(
                        /* [in] */ABI::Windows::Storage::IStorageFile * modelFile,
                        /* [out, retval] */__FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModel * * operation
                        ) = 0;
                    /* [overload, remote_async] */virtual HRESULT STDMETHODCALLTYPE LoadFromStreamAsync(
                        /* [in] */ABI::Windows::Storage::Streams::IRandomAccessStreamReference * modelStream,
                        /* [out, retval] */__FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModel * * operation
                        ) = 0;
                    /* [overload] */virtual HRESULT STDMETHODCALLTYPE LoadFromFilePath(
                        /* [in] */HSTRING filePath,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ILearningModel * * result
                        ) = 0;
                    /* [overload] */virtual HRESULT STDMETHODCALLTYPE LoadFromStream(
                        /* [in] */ABI::Windows::Storage::Streams::IRandomAccessStreamReference * modelStream,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ILearningModel * * result
                        ) = 0;
                    /* [overload, remote_async] */virtual HRESULT STDMETHODCALLTYPE LoadFromStorageFileWithOperatorProviderAsync(
                        /* [in] */ABI::Windows::Storage::IStorageFile * modelFile,
                        /* [in] */ABI::Microsoft::AI::MachineLearning::ILearningModelOperatorProvider * operatorProvider,
                        /* [out, retval] */__FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModel * * operation
                        ) = 0;
                    /* [overload, remote_async] */virtual HRESULT STDMETHODCALLTYPE LoadFromStreamWithOperatorProviderAsync(
                        /* [in] */ABI::Windows::Storage::Streams::IRandomAccessStreamReference * modelStream,
                        /* [in] */ABI::Microsoft::AI::MachineLearning::ILearningModelOperatorProvider * operatorProvider,
                        /* [out, retval] */__FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModel * * operation
                        ) = 0;
                    /* [overload] */virtual HRESULT STDMETHODCALLTYPE LoadFromFilePathWithOperatorProvider(
                        /* [in] */HSTRING filePath,
                        /* [in] */ABI::Microsoft::AI::MachineLearning::ILearningModelOperatorProvider * operatorProvider,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ILearningModel * * result
                        ) = 0;
                    /* [overload] */virtual HRESULT STDMETHODCALLTYPE LoadFromStreamWithOperatorProvider(
                        /* [in] */ABI::Windows::Storage::Streams::IRandomAccessStreamReference * modelStream,
                        /* [in] */ABI::Microsoft::AI::MachineLearning::ILearningModelOperatorProvider * operatorProvider,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ILearningModel * * result
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ILearningModelStatics=_uuidof(ILearningModelStatics);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Class Microsoft.AI.MachineLearning.LearningModel
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * RuntimeClass contains static methods.
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ILearningModelStatics interface starting with version 1.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.ILearningModel ** Default Interface **
 *    Windows.Foundation.IClosable
 *
 * Class Threading Model:  Both Single and Multi Threaded Apartment
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_LearningModel_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_LearningModel_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_LearningModel[] = L"Microsoft.AI.MachineLearning.LearningModel";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Struct Microsoft.AI.MachineLearning.LearningModelDeviceKind
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 */

#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [v1_enum, contract] */
                enum LearningModelDeviceKind : int
                {
                    LearningModelDeviceKind_Default = 0,
                    LearningModelDeviceKind_Cpu,
                    LearningModelDeviceKind_DirectX,
                    LearningModelDeviceKind_DirectXHighPerformance,
                    LearningModelDeviceKind_DirectXMinPower,
                };
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                class LearningModelDevice;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */



/*
 *
 * Interface Microsoft.AI.MachineLearning.ILearningModelDevice
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.LearningModelDevice
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ILearningModelDevice:HRESULT get_AdapterId(Windows.Graphics.DisplayAdapterId*);HRESULT get_Direct3D11Device(Windows.Graphics.DirectX.Direct3D11.IDirect3DDevice**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDevice_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDevice_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ILearningModelDevice[] = L"Microsoft.AI.MachineLearning.ILearningModelDevice";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("b56e79aa-89ee-5e25-8c5c-e32b68b51447"), contract, object, exclusiveto] */
                MIDL_INTERFACE("b56e79aa-89ee-5e25-8c5c-e32b68b51447")
                ILearningModelDevice : public IInspectable
                {
                public:
                    /* [propget] */virtual HRESULT STDMETHODCALLTYPE get_AdapterId(
                        /* [out, retval] */struct ABI::Windows::Graphics::DisplayAdapterId * value
                        ) = 0;
                    /* [propget] */virtual HRESULT STDMETHODCALLTYPE get_Direct3D11Device(
                        /* [out, retval] */ABI::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice * * value
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ILearningModelDevice=_uuidof(ILearningModelDevice);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDevice;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDevice_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ILearningModelDeviceStatics
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.LearningModelDevice
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ILearningModelDeviceStatics:HRESULT CreateFromDirect3D11Device(Windows.Graphics.DirectX.Direct3D11.IDirect3DDevice*,Microsoft.AI.MachineLearning.LearningModelDevice**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceStatics_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceStatics_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ILearningModelDeviceStatics[] = L"Microsoft.AI.MachineLearning.ILearningModelDeviceStatics";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("a622a3cd-0e4e-5f65-9a6f-06ab1b325582"), contract, object, exclusiveto] */
                MIDL_INTERFACE("a622a3cd-0e4e-5f65-9a6f-06ab1b325582")
                ILearningModelDeviceStatics : public IInspectable
                {
                public:
                    virtual HRESULT STDMETHODCALLTYPE CreateFromDirect3D11Device(
                        /* [in] */ABI::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice * device,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ILearningModelDevice * * result
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ILearningModelDeviceStatics=_uuidof(ILearningModelDeviceStatics);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceStatics;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceStatics_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ILearningModelDeviceFactory
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.LearningModelDevice
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ILearningModelDeviceFactory:HRESULT Create(Microsoft.AI.MachineLearning.LearningModelDeviceKind,Microsoft.AI.MachineLearning.LearningModelDevice**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceFactory_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceFactory_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ILearningModelDeviceFactory[] = L"Microsoft.AI.MachineLearning.ILearningModelDeviceFactory";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("0ad28f1e-8cdf-5dbc-8af3-51090e34e780"), contract, object, exclusiveto] */
                MIDL_INTERFACE("0ad28f1e-8cdf-5dbc-8af3-51090e34e780")
                ILearningModelDeviceFactory : public IInspectable
                {
                public:
                    virtual HRESULT STDMETHODCALLTYPE Create(
                        /* [in] */ABI::Microsoft::AI::MachineLearning::LearningModelDeviceKind deviceKind,
                        /* [retval, out] */ABI::Microsoft::AI::MachineLearning::ILearningModelDevice * * value
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ILearningModelDeviceFactory=_uuidof(ILearningModelDeviceFactory);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceFactory;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceFactory_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Class Microsoft.AI.MachineLearning.LearningModelDevice
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * RuntimeClass can be activated.
 *   Type can be activated via the Microsoft.AI.MachineLearning.ILearningModelDeviceFactory interface starting with version 1.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *
 * RuntimeClass contains static methods.
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ILearningModelDeviceStatics interface starting with version 1.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.ILearningModelDevice ** Default Interface **
 *
 * Class Threading Model:  Both Single and Multi Threaded Apartment
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_LearningModelDevice_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_LearningModelDevice_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_LearningModelDevice[] = L"Microsoft.AI.MachineLearning.LearningModelDevice";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000



/*
 *
 * Interface Microsoft.AI.MachineLearning.ILearningModelEvaluationResult
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.LearningModelEvaluationResult
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ILearningModelEvaluationResult:HRESULT get_CorrelationId(String*);HRESULT get_ErrorStatus(Int32*);HRESULT get_Succeeded(Boolean*);HRESULT get_Outputs(Windows.Foundation.Collections.IMapView`2<String, Object>**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelEvaluationResult_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelEvaluationResult_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ILearningModelEvaluationResult[] = L"Microsoft.AI.MachineLearning.ILearningModelEvaluationResult";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("576800d1-be6c-50ab-bd7e-a7e01508f3bc"), contract, object, exclusiveto] */
                MIDL_INTERFACE("576800d1-be6c-50ab-bd7e-a7e01508f3bc")
                ILearningModelEvaluationResult : public IInspectable
                {
                public:
                    /* [propget] */virtual HRESULT STDMETHODCALLTYPE get_CorrelationId(
                        /* [out, retval] */HSTRING * value
                        ) = 0;
                    /* [propget] */virtual HRESULT STDMETHODCALLTYPE get_ErrorStatus(
                        /* [out, retval] */int * value
                        ) = 0;
                    /* [propget] */virtual HRESULT STDMETHODCALLTYPE get_Succeeded(
                        /* [out, retval] */::boolean * value
                        ) = 0;
                    /* [propget] */virtual HRESULT STDMETHODCALLTYPE get_Outputs(
                        /* [out, retval] */__FIMapView_2_HSTRING_IInspectable * * value
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ILearningModelEvaluationResult=_uuidof(ILearningModelEvaluationResult);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelEvaluationResult;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelEvaluationResult_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Class Microsoft.AI.MachineLearning.LearningModelEvaluationResult
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.ILearningModelEvaluationResult ** Default Interface **
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_LearningModelEvaluationResult_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_LearningModelEvaluationResult_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_LearningModelEvaluationResult[] = L"Microsoft.AI.MachineLearning.LearningModelEvaluationResult";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                class LearningModelSessionOptions;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */



/*
 *
 * Interface Microsoft.AI.MachineLearning.ILearningModelSessionOptions
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 2.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.LearningModelSessionOptions
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ILearningModelSessionOptions:HRESULT get_BatchSizeOverride(UInt32*);HRESULT put_BatchSizeOverride(UInt32);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ILearningModelSessionOptions[] = L"Microsoft.AI.MachineLearning.ILearningModelSessionOptions";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("88968c30-4c69-5135-869a-26a0f4f9007f"), contract, object, exclusiveto] */
                MIDL_INTERFACE("88968c30-4c69-5135-869a-26a0f4f9007f")
                ILearningModelSessionOptions : public IInspectable
                {
                public:
                    /* [propget] */virtual HRESULT STDMETHODCALLTYPE get_BatchSizeOverride(
                        /* [out, retval] */unsigned int * value
                        ) = 0;
                    /* [propput] */virtual HRESULT STDMETHODCALLTYPE put_BatchSizeOverride(
                        /* [in] */unsigned int value
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ILearningModelSessionOptions=_uuidof(ILearningModelSessionOptions);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ILearningModelSessionOptions2
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 3.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.LearningModelSessionOptions
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ILearningModelSessionOptions2:HRESULT get_CloseModelOnSessionCreation(Boolean*);HRESULT put_CloseModelOnSessionCreation(Boolean);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x30000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions2_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions2_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ILearningModelSessionOptions2[] = L"Microsoft.AI.MachineLearning.ILearningModelSessionOptions2";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("46d6729a-e3a1-5412-a45b-d26460eda2db"), contract, object, exclusiveto] */
                MIDL_INTERFACE("46d6729a-e3a1-5412-a45b-d26460eda2db")
                ILearningModelSessionOptions2 : public IInspectable
                {
                public:
                    /* [propget] */virtual HRESULT STDMETHODCALLTYPE get_CloseModelOnSessionCreation(
                        /* [out, retval] */::boolean * value
                        ) = 0;
                    /* [propput] */virtual HRESULT STDMETHODCALLTYPE put_CloseModelOnSessionCreation(
                        /* [in] */::boolean value
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ILearningModelSessionOptions2=_uuidof(ILearningModelSessionOptions2);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions2;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions2_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x30000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ILearningModelSessionOptions3
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 4.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.LearningModelSessionOptions
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ILearningModelSessionOptions3:HRESULT OverrideNamedDimension(String,UInt32);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x40000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions3_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions3_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ILearningModelSessionOptions3[] = L"Microsoft.AI.MachineLearning.ILearningModelSessionOptions3";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("d346a34c-6639-5118-b38a-e3360ef1f69d"), contract, object, exclusiveto] */
                MIDL_INTERFACE("d346a34c-6639-5118-b38a-e3360ef1f69d")
                ILearningModelSessionOptions3 : public IInspectable
                {
                public:
                    virtual HRESULT STDMETHODCALLTYPE OverrideNamedDimension(
                        /* [in] */HSTRING name,
                        /* [in] */unsigned int dimension
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ILearningModelSessionOptions3=_uuidof(ILearningModelSessionOptions3);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions3;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions3_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x40000


/*
 *
 * Class Microsoft.AI.MachineLearning.LearningModelSessionOptions
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 2.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * RuntimeClass can be activated.
 *   Type can be activated via RoActivateInstance starting with version 2.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.ILearningModelSessionOptions ** Default Interface **
 *    Microsoft.AI.MachineLearning.ILearningModelSessionOptions2
 *    Microsoft.AI.MachineLearning.ILearningModelSessionOptions3
 *
 * Class Threading Model:  Both Single and Multi Threaded Apartment
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_LearningModelSessionOptions_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_LearningModelSessionOptions_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_LearningModelSessionOptions[] = L"Microsoft.AI.MachineLearning.LearningModelSessionOptions";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000

namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                class LearningModelSession;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */



/*
 *
 * Interface Microsoft.AI.MachineLearning.ILearningModelSession
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.LearningModelSession
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ILearningModelSession:HRESULT get_Model(Microsoft.AI.MachineLearning.LearningModel**);HRESULT get_Device(Microsoft.AI.MachineLearning.LearningModelDevice**);HRESULT get_EvaluationProperties(Windows.Foundation.Collections.IPropertySet**);HRESULT EvaluateAsync(Microsoft.AI.MachineLearning.LearningModelBinding*,String,Windows.Foundation.IAsyncOperation`1<Microsoft.AI.MachineLearning.LearningModelEvaluationResult>**);HRESULT EvaluateFeaturesAsync(Windows.Foundation.Collections.IMap`2<String, Object>*,String,Windows.Foundation.IAsyncOperation`1<Microsoft.AI.MachineLearning.LearningModelEvaluationResult>**);HRESULT Evaluate(Microsoft.AI.MachineLearning.LearningModelBinding*,String,Microsoft.AI.MachineLearning.LearningModelEvaluationResult**);HRESULT EvaluateFeatures(Windows.Foundation.Collections.IMap`2<String, Object>*,String,Microsoft.AI.MachineLearning.LearningModelEvaluationResult**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ILearningModelSession[] = L"Microsoft.AI.MachineLearning.ILearningModelSession";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("326031fc-c12d-5c0e-8209-4ad32fa41047"), contract, object, exclusiveto] */
                MIDL_INTERFACE("326031fc-c12d-5c0e-8209-4ad32fa41047")
                ILearningModelSession : public IInspectable
                {
                public:
                    /* [propget] */virtual HRESULT STDMETHODCALLTYPE get_Model(
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ILearningModel * * value
                        ) = 0;
                    /* [propget] */virtual HRESULT STDMETHODCALLTYPE get_Device(
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ILearningModelDevice * * value
                        ) = 0;
                    /* [propget] */virtual HRESULT STDMETHODCALLTYPE get_EvaluationProperties(
                        /* [out, retval] */ABI::Windows::Foundation::Collections::IPropertySet * * value
                        ) = 0;
                    /* [remote_async] */virtual HRESULT STDMETHODCALLTYPE EvaluateAsync(
                        /* [in] */ABI::Microsoft::AI::MachineLearning::ILearningModelBinding * bindings,
                        /* [in] */HSTRING correlationId,
                        /* [out, retval] */__FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult * * operation
                        ) = 0;
                    /* [remote_async] */virtual HRESULT STDMETHODCALLTYPE EvaluateFeaturesAsync(
                        /* [in] */__FIMap_2_HSTRING_IInspectable * features,
                        /* [in] */HSTRING correlationId,
                        /* [out, retval] */__FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult * * operation
                        ) = 0;
                    virtual HRESULT STDMETHODCALLTYPE Evaluate(
                        /* [in] */ABI::Microsoft::AI::MachineLearning::ILearningModelBinding * bindings,
                        /* [in] */HSTRING correlationId,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ILearningModelEvaluationResult * * result
                        ) = 0;
                    virtual HRESULT STDMETHODCALLTYPE EvaluateFeatures(
                        /* [in] */__FIMap_2_HSTRING_IInspectable * features,
                        /* [in] */HSTRING correlationId,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ILearningModelEvaluationResult * * result
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ILearningModelSession=_uuidof(ILearningModelSession);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ILearningModelSessionFactory
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.LearningModelSession
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ILearningModelSessionFactory:HRESULT CreateFromModel(Microsoft.AI.MachineLearning.LearningModel*,Microsoft.AI.MachineLearning.LearningModelSession**);HRESULT CreateFromModelOnDevice(Microsoft.AI.MachineLearning.LearningModel*,Microsoft.AI.MachineLearning.LearningModelDevice*,Microsoft.AI.MachineLearning.LearningModelSession**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ILearningModelSessionFactory[] = L"Microsoft.AI.MachineLearning.ILearningModelSessionFactory";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("d7d86c54-d03d-5ae3-a958-fe952b640620"), contract, object, exclusiveto] */
                MIDL_INTERFACE("d7d86c54-d03d-5ae3-a958-fe952b640620")
                ILearningModelSessionFactory : public IInspectable
                {
                public:
                    virtual HRESULT STDMETHODCALLTYPE CreateFromModel(
                        /* [in] */ABI::Microsoft::AI::MachineLearning::ILearningModel * model,
                        /* [retval, out] */ABI::Microsoft::AI::MachineLearning::ILearningModelSession * * value
                        ) = 0;
                    virtual HRESULT STDMETHODCALLTYPE CreateFromModelOnDevice(
                        /* [in] */ABI::Microsoft::AI::MachineLearning::ILearningModel * model,
                        /* [in] */ABI::Microsoft::AI::MachineLearning::ILearningModelDevice * deviceToRunOn,
                        /* [retval, out] */ABI::Microsoft::AI::MachineLearning::ILearningModelSession * * value
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ILearningModelSessionFactory=_uuidof(ILearningModelSessionFactory);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ILearningModelSessionFactory2
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 2.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.LearningModelSession
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ILearningModelSessionFactory2:HRESULT CreateFromModelOnDeviceWithSessionOptions(Microsoft.AI.MachineLearning.LearningModel*,Microsoft.AI.MachineLearning.LearningModelDevice*,Microsoft.AI.MachineLearning.LearningModelSessionOptions*,Microsoft.AI.MachineLearning.LearningModelSession**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory2_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory2_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ILearningModelSessionFactory2[] = L"Microsoft.AI.MachineLearning.ILearningModelSessionFactory2";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("1a25cd87-032c-5775-bdaa-852915a025a9"), contract, object, exclusiveto] */
                MIDL_INTERFACE("1a25cd87-032c-5775-bdaa-852915a025a9")
                ILearningModelSessionFactory2 : public IInspectable
                {
                public:
                    virtual HRESULT STDMETHODCALLTYPE CreateFromModelOnDeviceWithSessionOptions(
                        /* [in] */ABI::Microsoft::AI::MachineLearning::ILearningModel * model,
                        /* [in] */ABI::Microsoft::AI::MachineLearning::ILearningModelDevice * deviceToRunOn,
                        /* [in] */ABI::Microsoft::AI::MachineLearning::ILearningModelSessionOptions * learningModelSessionOptions,
                        /* [retval, out] */ABI::Microsoft::AI::MachineLearning::ILearningModelSession * * value
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ILearningModelSessionFactory2=_uuidof(ILearningModelSessionFactory2);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory2;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory2_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000


/*
 *
 * Class Microsoft.AI.MachineLearning.LearningModelSession
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * RuntimeClass can be activated.
 *   Type can be activated via the Microsoft.AI.MachineLearning.ILearningModelSessionFactory2 interface starting with version 2.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *   Type can be activated via the Microsoft.AI.MachineLearning.ILearningModelSessionFactory interface starting with version 1.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.ILearningModelSession ** Default Interface **
 *    Windows.Foundation.IClosable
 *
 * Class Threading Model:  Both Single and Multi Threaded Apartment
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_LearningModelSession_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_LearningModelSession_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_LearningModelSession[] = L"Microsoft.AI.MachineLearning.LearningModelSession";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ILearningModelFeatureValue
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ILearningModelFeatureValue:HRESULT get_Kind(Microsoft.AI.MachineLearning.LearningModelFeatureKind*);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureValue_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureValue_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ILearningModelFeatureValue[] = L"Microsoft.AI.MachineLearning.ILearningModelFeatureValue";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("ae7816ef-cb63-5aa4-81fc-e902a0a6c977"), object, contract] */
                MIDL_INTERFACE("ae7816ef-cb63-5aa4-81fc-e902a0a6c977")
                ILearningModelFeatureValue : public IInspectable
                {
                public:
                    /* [propget] */virtual HRESULT STDMETHODCALLTYPE get_Kind(
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::LearningModelFeatureKind * value
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ILearningModelFeatureValue=_uuidof(ILearningModelFeatureValue);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureValue;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureValue_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ILearningModelBinding
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.LearningModelBinding
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ILearningModelBinding:HRESULT Bind(String,Object*);HRESULT BindWithProperties(String,Object*,Windows.Foundation.Collections.IPropertySet*);HRESULT Clear();
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBinding_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBinding_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ILearningModelBinding[] = L"Microsoft.AI.MachineLearning.ILearningModelBinding";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("2f77b18a-68ae-510f-afb9-5af57446c343"), contract, object, exclusiveto] */
                MIDL_INTERFACE("2f77b18a-68ae-510f-afb9-5af57446c343")
                ILearningModelBinding : public IInspectable
                {
                public:
                    /* [overload] */virtual HRESULT STDMETHODCALLTYPE Bind(
                        /* [in] */HSTRING name,
                        /* [in] */IInspectable * value
                        ) = 0;
                    /* [overload] */virtual HRESULT STDMETHODCALLTYPE BindWithProperties(
                        /* [in] */HSTRING name,
                        /* [in] */IInspectable * value,
                        /* [in] */ABI::Windows::Foundation::Collections::IPropertySet * props
                        ) = 0;
                    virtual HRESULT STDMETHODCALLTYPE Clear(void) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ILearningModelBinding=_uuidof(ILearningModelBinding);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBinding;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBinding_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ILearningModelBindingFactory
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.LearningModelBinding
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ILearningModelBindingFactory:HRESULT CreateFromSession(Microsoft.AI.MachineLearning.LearningModelSession*,Microsoft.AI.MachineLearning.LearningModelBinding**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBindingFactory_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBindingFactory_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ILearningModelBindingFactory[] = L"Microsoft.AI.MachineLearning.ILearningModelBindingFactory";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("ae2f1c97-2fd5-55b9-a05f-53b9dbb4f9e2"), contract, object, exclusiveto] */
                MIDL_INTERFACE("ae2f1c97-2fd5-55b9-a05f-53b9dbb4f9e2")
                ILearningModelBindingFactory : public IInspectable
                {
                public:
                    virtual HRESULT STDMETHODCALLTYPE CreateFromSession(
                        /* [in] */ABI::Microsoft::AI::MachineLearning::ILearningModelSession * session,
                        /* [retval, out] */ABI::Microsoft::AI::MachineLearning::ILearningModelBinding * * value
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ILearningModelBindingFactory=_uuidof(ILearningModelBindingFactory);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBindingFactory;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBindingFactory_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Class Microsoft.AI.MachineLearning.LearningModelBinding
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * RuntimeClass can be activated.
 *   Type can be activated via the Microsoft.AI.MachineLearning.ILearningModelBindingFactory interface starting with version 1.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.ILearningModelBinding ** Default Interface **
 *    Windows.Foundation.Collections.IMapView_2_HSTRING,IInspectable
 *
 * Class Threading Model:  Both Single and Multi Threaded Apartment
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_LearningModelBinding_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_LearningModelBinding_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_LearningModelBinding[] = L"Microsoft.AI.MachineLearning.LearningModelBinding";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Struct Microsoft.AI.MachineLearning.TensorKind
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 */

#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [v1_enum, contract] */
                enum TensorKind : int
                {
                    TensorKind_Undefined = 0,
                    TensorKind_Float,
                    TensorKind_UInt8,
                    TensorKind_Int8,
                    TensorKind_UInt16,
                    TensorKind_Int16,
                    TensorKind_Int32,
                    TensorKind_Int64,
                    TensorKind_String,
                    TensorKind_Boolean,
                    TensorKind_Float16,
                    TensorKind_Double,
                    TensorKind_UInt32,
                    TensorKind_UInt64,
                    TensorKind_Complex64,
                    TensorKind_Complex128,
                };
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                class MapFeatureDescriptor;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */



/*
 *
 * Interface Microsoft.AI.MachineLearning.IMapFeatureDescriptor
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.MapFeatureDescriptor
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.IMapFeatureDescriptor:HRESULT get_KeyKind(Microsoft.AI.MachineLearning.TensorKind*);HRESULT get_ValueDescriptor(Microsoft.AI.MachineLearning.ILearningModelFeatureDescriptor**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CIMapFeatureDescriptor_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CIMapFeatureDescriptor_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_IMapFeatureDescriptor[] = L"Microsoft.AI.MachineLearning.IMapFeatureDescriptor";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("308161e6-8c55-5db3-8ea7-b70edae7f362"), contract, object, exclusiveto] */
                MIDL_INTERFACE("308161e6-8c55-5db3-8ea7-b70edae7f362")
                IMapFeatureDescriptor : public IInspectable
                {
                public:
                    /* [propget] */virtual HRESULT STDMETHODCALLTYPE get_KeyKind(
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::TensorKind * value
                        ) = 0;
                    /* [propget] */virtual HRESULT STDMETHODCALLTYPE get_ValueDescriptor(
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ILearningModelFeatureDescriptor * * value
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_IMapFeatureDescriptor=_uuidof(IMapFeatureDescriptor);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CIMapFeatureDescriptor;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CIMapFeatureDescriptor_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Class Microsoft.AI.MachineLearning.MapFeatureDescriptor
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.IMapFeatureDescriptor ** Default Interface **
 *    Microsoft.AI.MachineLearning.ILearningModelFeatureDescriptor
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_MapFeatureDescriptor_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_MapFeatureDescriptor_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_MapFeatureDescriptor[] = L"Microsoft.AI.MachineLearning.MapFeatureDescriptor";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                class SequenceFeatureDescriptor;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */



/*
 *
 * Interface Microsoft.AI.MachineLearning.ISequenceFeatureDescriptor
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.SequenceFeatureDescriptor
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ISequenceFeatureDescriptor:HRESULT get_ElementDescriptor(Microsoft.AI.MachineLearning.ILearningModelFeatureDescriptor**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CISequenceFeatureDescriptor_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CISequenceFeatureDescriptor_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ISequenceFeatureDescriptor[] = L"Microsoft.AI.MachineLearning.ISequenceFeatureDescriptor";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("bcc21461-ba27-54f0-a5c6-2bb81aca668d"), contract, object, exclusiveto] */
                MIDL_INTERFACE("bcc21461-ba27-54f0-a5c6-2bb81aca668d")
                ISequenceFeatureDescriptor : public IInspectable
                {
                public:
                    /* [propget] */virtual HRESULT STDMETHODCALLTYPE get_ElementDescriptor(
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ILearningModelFeatureDescriptor * * value
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ISequenceFeatureDescriptor=_uuidof(ISequenceFeatureDescriptor);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CISequenceFeatureDescriptor;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CISequenceFeatureDescriptor_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Class Microsoft.AI.MachineLearning.SequenceFeatureDescriptor
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.ISequenceFeatureDescriptor ** Default Interface **
 *    Microsoft.AI.MachineLearning.ILearningModelFeatureDescriptor
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_SequenceFeatureDescriptor_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_SequenceFeatureDescriptor_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_SequenceFeatureDescriptor[] = L"Microsoft.AI.MachineLearning.SequenceFeatureDescriptor";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                class TensorFeatureDescriptor;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */



/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorFeatureDescriptor
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorFeatureDescriptor
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorFeatureDescriptor:HRESULT get_TensorKind(Microsoft.AI.MachineLearning.TensorKind*);HRESULT get_Shape(Windows.Foundation.Collections.IVectorView`1<Int64>**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFeatureDescriptor_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFeatureDescriptor_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorFeatureDescriptor[] = L"Microsoft.AI.MachineLearning.ITensorFeatureDescriptor";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("4de2edbe-6519-51ef-b88b-8d9820449e1b"), contract, object, exclusiveto] */
                MIDL_INTERFACE("4de2edbe-6519-51ef-b88b-8d9820449e1b")
                ITensorFeatureDescriptor : public IInspectable
                {
                public:
                    /* [propget] */virtual HRESULT STDMETHODCALLTYPE get_TensorKind(
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::TensorKind * value
                        ) = 0;
                    /* [propget] */virtual HRESULT STDMETHODCALLTYPE get_Shape(
                        /* [out, retval] */__FIVectorView_1___z__zint64 * * value
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ITensorFeatureDescriptor=_uuidof(ITensorFeatureDescriptor);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFeatureDescriptor;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFeatureDescriptor_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Class Microsoft.AI.MachineLearning.TensorFeatureDescriptor
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.ITensorFeatureDescriptor ** Default Interface **
 *    Microsoft.AI.MachineLearning.ILearningModelFeatureDescriptor
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorFeatureDescriptor_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorFeatureDescriptor_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_TensorFeatureDescriptor[] = L"Microsoft.AI.MachineLearning.TensorFeatureDescriptor";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Struct Microsoft.AI.MachineLearning.LearningModelPixelRange
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 5.0
 *
 *
 */

#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x50000
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [v1_enum, contract] */
                enum LearningModelPixelRange : int
                {
                    LearningModelPixelRange_ZeroTo255 = 0,
                    LearningModelPixelRange_ZeroToOne,
                    LearningModelPixelRange_MinusOneToOne,
                };
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x50000

namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                class ImageFeatureDescriptor;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */



/*
 *
 * Interface Microsoft.AI.MachineLearning.IImageFeatureDescriptor
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.ImageFeatureDescriptor
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.IImageFeatureDescriptor:HRESULT get_BitmapPixelFormat(Windows.Graphics.Imaging.BitmapPixelFormat*);HRESULT get_BitmapAlphaMode(Windows.Graphics.Imaging.BitmapAlphaMode*);HRESULT get_Width(UInt32*);HRESULT get_Height(UInt32*);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_IImageFeatureDescriptor[] = L"Microsoft.AI.MachineLearning.IImageFeatureDescriptor";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("5c9c1823-77b7-5b43-aafc-9c540c50ff60"), contract, object, exclusiveto] */
                MIDL_INTERFACE("5c9c1823-77b7-5b43-aafc-9c540c50ff60")
                IImageFeatureDescriptor : public IInspectable
                {
                public:
                    /* [propget] */virtual HRESULT STDMETHODCALLTYPE get_BitmapPixelFormat(
                        /* [out, retval] */ABI::Windows::Graphics::Imaging::BitmapPixelFormat * value
                        ) = 0;
                    /* [propget] */virtual HRESULT STDMETHODCALLTYPE get_BitmapAlphaMode(
                        /* [out, retval] */ABI::Windows::Graphics::Imaging::BitmapAlphaMode * value
                        ) = 0;
                    /* [propget] */virtual HRESULT STDMETHODCALLTYPE get_Width(
                        /* [out, retval] */unsigned int * value
                        ) = 0;
                    /* [propget] */virtual HRESULT STDMETHODCALLTYPE get_Height(
                        /* [out, retval] */unsigned int * value
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_IImageFeatureDescriptor=_uuidof(IImageFeatureDescriptor);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.IImageFeatureDescriptor2
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 5.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.ImageFeatureDescriptor
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.IImageFeatureDescriptor2:HRESULT get_PixelRange(Microsoft.AI.MachineLearning.LearningModelPixelRange*);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x50000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor2_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor2_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_IImageFeatureDescriptor2[] = L"Microsoft.AI.MachineLearning.IImageFeatureDescriptor2";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("01f87bfc-90b9-5af7-8cbd-aa9e0e96f4d2"), contract, object, exclusiveto] */
                MIDL_INTERFACE("01f87bfc-90b9-5af7-8cbd-aa9e0e96f4d2")
                IImageFeatureDescriptor2 : public IInspectable
                {
                public:
                    /* [propget] */virtual HRESULT STDMETHODCALLTYPE get_PixelRange(
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::LearningModelPixelRange * value
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_IImageFeatureDescriptor2=_uuidof(IImageFeatureDescriptor2);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor2;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor2_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x50000


/*
 *
 * Class Microsoft.AI.MachineLearning.ImageFeatureDescriptor
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.IImageFeatureDescriptor ** Default Interface **
 *    Microsoft.AI.MachineLearning.IImageFeatureDescriptor2
 *    Microsoft.AI.MachineLearning.ILearningModelFeatureDescriptor
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_ImageFeatureDescriptor_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_ImageFeatureDescriptor_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_ImageFeatureDescriptor[] = L"Microsoft.AI.MachineLearning.ImageFeatureDescriptor";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensor
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Any object which implements this interface must also implement the following interfaces:
 *     Microsoft.AI.MachineLearning.ILearningModelFeatureValue
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensor:HRESULT get_TensorKind(Microsoft.AI.MachineLearning.TensorKind*);HRESULT get_Shape(Windows.Foundation.Collections.IVectorView`1<Int64>**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensor_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensor_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensor[] = L"Microsoft.AI.MachineLearning.ITensor";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("376f95b3-eb6c-551f-b76b-369e925b4149"), object, contract] */
                MIDL_INTERFACE("376f95b3-eb6c-551f-b76b-369e925b4149")
                ITensor : public IInspectable
                {
                public:
                    /* [propget] */virtual HRESULT STDMETHODCALLTYPE get_TensorKind(
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::TensorKind * value
                        ) = 0;
                    /* [propget] */virtual HRESULT STDMETHODCALLTYPE get_Shape(
                        /* [out, retval] */__FIVectorView_1___z__zint64 * * value
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ITensor=_uuidof(ITensor);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensor;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensor_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                class TensorFloat;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */



/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorFloat
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorFloat
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorFloat:HRESULT GetAsVectorView(Windows.Foundation.Collections.IVectorView`1<Single>**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorFloat[] = L"Microsoft.AI.MachineLearning.ITensorFloat";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("b137fe61-183a-51b1-87de-f5662d9e5f34"), contract, object, exclusiveto] */
                MIDL_INTERFACE("b137fe61-183a-51b1-87de-f5662d9e5f34")
                ITensorFloat : public IInspectable
                {
                public:
                    virtual HRESULT STDMETHODCALLTYPE GetAsVectorView(
                        /* [out, retval] */__FIVectorView_1_float * * result
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ITensorFloat=_uuidof(ITensorFloat);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorFloatStatics
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorFloat
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorFloatStatics:HRESULT Create(Microsoft.AI.MachineLearning.TensorFloat**);HRESULT Create2(Windows.Foundation.Collections.IIterable`1<Int64>*,Microsoft.AI.MachineLearning.TensorFloat**);HRESULT CreateFromArray(Windows.Foundation.Collections.IIterable`1<Int64>*,UInt32,Single*,Microsoft.AI.MachineLearning.TensorFloat**);HRESULT CreateFromIterable(Windows.Foundation.Collections.IIterable`1<Int64>*,Windows.Foundation.Collections.IIterable`1<Single>*,Microsoft.AI.MachineLearning.TensorFloat**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorFloatStatics[] = L"Microsoft.AI.MachineLearning.ITensorFloatStatics";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("650fee2e-c9f8-54bd-a895-4d37e2e146d0"), contract, object, exclusiveto] */
                MIDL_INTERFACE("650fee2e-c9f8-54bd-a895-4d37e2e146d0")
                ITensorFloatStatics : public IInspectable
                {
                public:
                    /* [overload] */virtual HRESULT STDMETHODCALLTYPE Create(
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorFloat * * result
                        ) = 0;
                    /* [overload] */virtual HRESULT STDMETHODCALLTYPE Create2(
                        /* [in] */__FIIterable_1___z__zint64 * shape,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorFloat * * result
                        ) = 0;
                    virtual HRESULT STDMETHODCALLTYPE CreateFromArray(
                        /* [in] */__FIIterable_1___z__zint64 * shape,
                        /* [in] */unsigned int dataLength,
                        /* [size_is(dataLength), in] */float * data,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorFloat * * result
                        ) = 0;
                    virtual HRESULT STDMETHODCALLTYPE CreateFromIterable(
                        /* [in] */__FIIterable_1___z__zint64 * shape,
                        /* [in] */__FIIterable_1_float * data,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorFloat * * result
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ITensorFloatStatics=_uuidof(ITensorFloatStatics);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorFloatStatics2
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 2.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorFloat
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorFloatStatics2:HRESULT CreateFromShapeArrayAndDataArray(UInt32,Int64*,UInt32,Single*,Microsoft.AI.MachineLearning.TensorFloat**);HRESULT CreateFromBuffer(UInt32,Int64*,Windows.Storage.Streams.IBuffer*,Microsoft.AI.MachineLearning.TensorFloat**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics2_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics2_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorFloatStatics2[] = L"Microsoft.AI.MachineLearning.ITensorFloatStatics2";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("73e9f04f-5f05-515a-8179-8ffe46a639cf"), contract, object, exclusiveto] */
                MIDL_INTERFACE("73e9f04f-5f05-515a-8179-8ffe46a639cf")
                ITensorFloatStatics2 : public IInspectable
                {
                public:
                    virtual HRESULT STDMETHODCALLTYPE CreateFromShapeArrayAndDataArray(
                        /* [in] */unsigned int shapeLength,
                        /* [size_is(shapeLength), in] */__int64 * shape,
                        /* [in] */unsigned int dataLength,
                        /* [size_is(dataLength), in] */float * data,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorFloat * * result
                        ) = 0;
                    virtual HRESULT STDMETHODCALLTYPE CreateFromBuffer(
                        /* [in] */unsigned int shapeLength,
                        /* [size_is(shapeLength), in] */__int64 * shape,
                        /* [in] */ABI::Windows::Storage::Streams::IBuffer * buffer,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorFloat * * result
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ITensorFloatStatics2=_uuidof(ITensorFloatStatics2);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics2;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics2_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000


/*
 *
 * Class Microsoft.AI.MachineLearning.TensorFloat
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * RuntimeClass contains static methods.
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorFloatStatics2 interface starting with version 2.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorFloatStatics interface starting with version 1.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.ITensorFloat ** Default Interface **
 *    Microsoft.AI.MachineLearning.ITensor
 *    Windows.Foundation.IMemoryBuffer
 *
 * Class Threading Model:  Both Single and Multi Threaded Apartment
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorFloat_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorFloat_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_TensorFloat[] = L"Microsoft.AI.MachineLearning.TensorFloat";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                class TensorFloat16Bit;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */



/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorFloat16Bit
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorFloat16Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorFloat16Bit:HRESULT GetAsVectorView(Windows.Foundation.Collections.IVectorView`1<Single>**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16Bit_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16Bit_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorFloat16Bit[] = L"Microsoft.AI.MachineLearning.ITensorFloat16Bit";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("5b653d83-e84f-59bc-a489-014ad5c332a2"), contract, object, exclusiveto] */
                MIDL_INTERFACE("5b653d83-e84f-59bc-a489-014ad5c332a2")
                ITensorFloat16Bit : public IInspectable
                {
                public:
                    virtual HRESULT STDMETHODCALLTYPE GetAsVectorView(
                        /* [out, retval] */__FIVectorView_1_float * * result
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ITensorFloat16Bit=_uuidof(ITensorFloat16Bit);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16Bit;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16Bit_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorFloat16BitStatics
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorFloat16Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorFloat16BitStatics:HRESULT Create(Microsoft.AI.MachineLearning.TensorFloat16Bit**);HRESULT Create2(Windows.Foundation.Collections.IIterable`1<Int64>*,Microsoft.AI.MachineLearning.TensorFloat16Bit**);HRESULT CreateFromArray(Windows.Foundation.Collections.IIterable`1<Int64>*,UInt32,Single*,Microsoft.AI.MachineLearning.TensorFloat16Bit**);HRESULT CreateFromIterable(Windows.Foundation.Collections.IIterable`1<Int64>*,Windows.Foundation.Collections.IIterable`1<Single>*,Microsoft.AI.MachineLearning.TensorFloat16Bit**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorFloat16BitStatics[] = L"Microsoft.AI.MachineLearning.ITensorFloat16BitStatics";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("87dfb838-eb96-5d5e-bb03-efab0b6a7b0d"), contract, object, exclusiveto] */
                MIDL_INTERFACE("87dfb838-eb96-5d5e-bb03-efab0b6a7b0d")
                ITensorFloat16BitStatics : public IInspectable
                {
                public:
                    /* [overload] */virtual HRESULT STDMETHODCALLTYPE Create(
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorFloat16Bit * * result
                        ) = 0;
                    /* [overload] */virtual HRESULT STDMETHODCALLTYPE Create2(
                        /* [in] */__FIIterable_1___z__zint64 * shape,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorFloat16Bit * * result
                        ) = 0;
                    virtual HRESULT STDMETHODCALLTYPE CreateFromArray(
                        /* [in] */__FIIterable_1___z__zint64 * shape,
                        /* [in] */unsigned int dataLength,
                        /* [size_is(dataLength), in] */float * data,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorFloat16Bit * * result
                        ) = 0;
                    virtual HRESULT STDMETHODCALLTYPE CreateFromIterable(
                        /* [in] */__FIIterable_1___z__zint64 * shape,
                        /* [in] */__FIIterable_1_float * data,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorFloat16Bit * * result
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ITensorFloat16BitStatics=_uuidof(ITensorFloat16BitStatics);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorFloat16BitStatics2
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 2.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorFloat16Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorFloat16BitStatics2:HRESULT CreateFromShapeArrayAndDataArray(UInt32,Int64*,UInt32,Single*,Microsoft.AI.MachineLearning.TensorFloat16Bit**);HRESULT CreateFromBuffer(UInt32,Int64*,Windows.Storage.Streams.IBuffer*,Microsoft.AI.MachineLearning.TensorFloat16Bit**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics2_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics2_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorFloat16BitStatics2[] = L"Microsoft.AI.MachineLearning.ITensorFloat16BitStatics2";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("4af7175c-b626-50a3-a169-555de125bc2d"), contract, object, exclusiveto] */
                MIDL_INTERFACE("4af7175c-b626-50a3-a169-555de125bc2d")
                ITensorFloat16BitStatics2 : public IInspectable
                {
                public:
                    virtual HRESULT STDMETHODCALLTYPE CreateFromShapeArrayAndDataArray(
                        /* [in] */unsigned int shapeLength,
                        /* [size_is(shapeLength), in] */__int64 * shape,
                        /* [in] */unsigned int dataLength,
                        /* [size_is(dataLength), in] */float * data,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorFloat16Bit * * result
                        ) = 0;
                    virtual HRESULT STDMETHODCALLTYPE CreateFromBuffer(
                        /* [in] */unsigned int shapeLength,
                        /* [size_is(shapeLength), in] */__int64 * shape,
                        /* [in] */ABI::Windows::Storage::Streams::IBuffer * buffer,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorFloat16Bit * * result
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ITensorFloat16BitStatics2=_uuidof(ITensorFloat16BitStatics2);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics2;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics2_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000


/*
 *
 * Class Microsoft.AI.MachineLearning.TensorFloat16Bit
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * RuntimeClass contains static methods.
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorFloat16BitStatics2 interface starting with version 2.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorFloat16BitStatics interface starting with version 1.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.ITensorFloat16Bit ** Default Interface **
 *    Microsoft.AI.MachineLearning.ITensor
 *    Windows.Foundation.IMemoryBuffer
 *
 * Class Threading Model:  Both Single and Multi Threaded Apartment
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorFloat16Bit_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorFloat16Bit_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_TensorFloat16Bit[] = L"Microsoft.AI.MachineLearning.TensorFloat16Bit";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                class TensorUInt8Bit;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */



/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorUInt8Bit
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorUInt8Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorUInt8Bit:HRESULT GetAsVectorView(Windows.Foundation.Collections.IVectorView`1<UInt8>**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8Bit_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8Bit_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorUInt8Bit[] = L"Microsoft.AI.MachineLearning.ITensorUInt8Bit";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("9156f628-73c4-52b7-931e-0f010c06d5ee"), contract, object, exclusiveto] */
                MIDL_INTERFACE("9156f628-73c4-52b7-931e-0f010c06d5ee")
                ITensorUInt8Bit : public IInspectable
                {
                public:
                    virtual HRESULT STDMETHODCALLTYPE GetAsVectorView(
                        /* [out, retval] */__FIVectorView_1_byte * * result
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ITensorUInt8Bit=_uuidof(ITensorUInt8Bit);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8Bit;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8Bit_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorUInt8BitStatics
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorUInt8Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorUInt8BitStatics:HRESULT Create(Microsoft.AI.MachineLearning.TensorUInt8Bit**);HRESULT Create2(Windows.Foundation.Collections.IIterable`1<Int64>*,Microsoft.AI.MachineLearning.TensorUInt8Bit**);HRESULT CreateFromArray(Windows.Foundation.Collections.IIterable`1<Int64>*,UInt32,UInt8*,Microsoft.AI.MachineLearning.TensorUInt8Bit**);HRESULT CreateFromIterable(Windows.Foundation.Collections.IIterable`1<Int64>*,Windows.Foundation.Collections.IIterable`1<UInt8>*,Microsoft.AI.MachineLearning.TensorUInt8Bit**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorUInt8BitStatics[] = L"Microsoft.AI.MachineLearning.ITensorUInt8BitStatics";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("977e8827-fee9-57c3-a7b1-0aa1c3bc6457"), contract, object, exclusiveto] */
                MIDL_INTERFACE("977e8827-fee9-57c3-a7b1-0aa1c3bc6457")
                ITensorUInt8BitStatics : public IInspectable
                {
                public:
                    /* [overload] */virtual HRESULT STDMETHODCALLTYPE Create(
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorUInt8Bit * * result
                        ) = 0;
                    /* [overload] */virtual HRESULT STDMETHODCALLTYPE Create2(
                        /* [in] */__FIIterable_1___z__zint64 * shape,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorUInt8Bit * * result
                        ) = 0;
                    virtual HRESULT STDMETHODCALLTYPE CreateFromArray(
                        /* [in] */__FIIterable_1___z__zint64 * shape,
                        /* [in] */unsigned int dataLength,
                        /* [size_is(dataLength), in] */::byte * data,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorUInt8Bit * * result
                        ) = 0;
                    virtual HRESULT STDMETHODCALLTYPE CreateFromIterable(
                        /* [in] */__FIIterable_1___z__zint64 * shape,
                        /* [in] */__FIIterable_1_byte * data,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorUInt8Bit * * result
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ITensorUInt8BitStatics=_uuidof(ITensorUInt8BitStatics);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorUInt8BitStatics2
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 2.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorUInt8Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorUInt8BitStatics2:HRESULT CreateFromShapeArrayAndDataArray(UInt32,Int64*,UInt32,UInt8*,Microsoft.AI.MachineLearning.TensorUInt8Bit**);HRESULT CreateFromBuffer(UInt32,Int64*,Windows.Storage.Streams.IBuffer*,Microsoft.AI.MachineLearning.TensorUInt8Bit**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics2_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics2_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorUInt8BitStatics2[] = L"Microsoft.AI.MachineLearning.ITensorUInt8BitStatics2";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("790b9742-c09f-584a-910d-9f5197cbecf4"), contract, object, exclusiveto] */
                MIDL_INTERFACE("790b9742-c09f-584a-910d-9f5197cbecf4")
                ITensorUInt8BitStatics2 : public IInspectable
                {
                public:
                    virtual HRESULT STDMETHODCALLTYPE CreateFromShapeArrayAndDataArray(
                        /* [in] */unsigned int shapeLength,
                        /* [size_is(shapeLength), in] */__int64 * shape,
                        /* [in] */unsigned int dataLength,
                        /* [size_is(dataLength), in] */::byte * data,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorUInt8Bit * * result
                        ) = 0;
                    virtual HRESULT STDMETHODCALLTYPE CreateFromBuffer(
                        /* [in] */unsigned int shapeLength,
                        /* [size_is(shapeLength), in] */__int64 * shape,
                        /* [in] */ABI::Windows::Storage::Streams::IBuffer * buffer,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorUInt8Bit * * result
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ITensorUInt8BitStatics2=_uuidof(ITensorUInt8BitStatics2);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics2;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics2_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000


/*
 *
 * Class Microsoft.AI.MachineLearning.TensorUInt8Bit
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * RuntimeClass contains static methods.
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorUInt8BitStatics2 interface starting with version 2.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorUInt8BitStatics interface starting with version 1.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.ITensorUInt8Bit ** Default Interface **
 *    Microsoft.AI.MachineLearning.ITensor
 *    Windows.Foundation.IMemoryBuffer
 *
 * Class Threading Model:  Both Single and Multi Threaded Apartment
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorUInt8Bit_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorUInt8Bit_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_TensorUInt8Bit[] = L"Microsoft.AI.MachineLearning.TensorUInt8Bit";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                class TensorInt8Bit;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */



/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorInt8Bit
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorInt8Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorInt8Bit:HRESULT GetAsVectorView(Windows.Foundation.Collections.IVectorView`1<UInt8>**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8Bit_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8Bit_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorInt8Bit[] = L"Microsoft.AI.MachineLearning.ITensorInt8Bit";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("c7cfa5d8-b290-5310-bf7c-cbf92c9de764"), contract, object, exclusiveto] */
                MIDL_INTERFACE("c7cfa5d8-b290-5310-bf7c-cbf92c9de764")
                ITensorInt8Bit : public IInspectable
                {
                public:
                    virtual HRESULT STDMETHODCALLTYPE GetAsVectorView(
                        /* [out, retval] */__FIVectorView_1_byte * * result
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ITensorInt8Bit=_uuidof(ITensorInt8Bit);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8Bit;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8Bit_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorInt8BitStatics
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorInt8Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorInt8BitStatics:HRESULT Create(Microsoft.AI.MachineLearning.TensorInt8Bit**);HRESULT Create2(Windows.Foundation.Collections.IIterable`1<Int64>*,Microsoft.AI.MachineLearning.TensorInt8Bit**);HRESULT CreateFromArray(Windows.Foundation.Collections.IIterable`1<Int64>*,UInt32,UInt8*,Microsoft.AI.MachineLearning.TensorInt8Bit**);HRESULT CreateFromIterable(Windows.Foundation.Collections.IIterable`1<Int64>*,Windows.Foundation.Collections.IIterable`1<UInt8>*,Microsoft.AI.MachineLearning.TensorInt8Bit**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorInt8BitStatics[] = L"Microsoft.AI.MachineLearning.ITensorInt8BitStatics";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("3da5e2b3-4cf6-5d6f-84d3-0b168bd9cba0"), contract, object, exclusiveto] */
                MIDL_INTERFACE("3da5e2b3-4cf6-5d6f-84d3-0b168bd9cba0")
                ITensorInt8BitStatics : public IInspectable
                {
                public:
                    /* [overload] */virtual HRESULT STDMETHODCALLTYPE Create(
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorInt8Bit * * result
                        ) = 0;
                    /* [overload] */virtual HRESULT STDMETHODCALLTYPE Create2(
                        /* [in] */__FIIterable_1___z__zint64 * shape,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorInt8Bit * * result
                        ) = 0;
                    virtual HRESULT STDMETHODCALLTYPE CreateFromArray(
                        /* [in] */__FIIterable_1___z__zint64 * shape,
                        /* [in] */unsigned int dataLength,
                        /* [size_is(dataLength), in] */BYTE * data,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorInt8Bit * * result
                        ) = 0;
                    virtual HRESULT STDMETHODCALLTYPE CreateFromIterable(
                        /* [in] */__FIIterable_1___z__zint64 * shape,
                        /* [in] */__FIIterable_1_byte * data,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorInt8Bit * * result
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ITensorInt8BitStatics=_uuidof(ITensorInt8BitStatics);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorInt8BitStatics2
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 2.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorInt8Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorInt8BitStatics2:HRESULT CreateFromShapeArrayAndDataArray(UInt32,Int64*,UInt32,UInt8*,Microsoft.AI.MachineLearning.TensorInt8Bit**);HRESULT CreateFromBuffer(UInt32,Int64*,Windows.Storage.Streams.IBuffer*,Microsoft.AI.MachineLearning.TensorInt8Bit**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics2_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics2_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorInt8BitStatics2[] = L"Microsoft.AI.MachineLearning.ITensorInt8BitStatics2";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("1cb2caa5-352e-5984-8f43-7e4c611b0e52"), contract, object, exclusiveto] */
                MIDL_INTERFACE("1cb2caa5-352e-5984-8f43-7e4c611b0e52")
                ITensorInt8BitStatics2 : public IInspectable
                {
                public:
                    virtual HRESULT STDMETHODCALLTYPE CreateFromShapeArrayAndDataArray(
                        /* [in] */unsigned int shapeLength,
                        /* [size_is(shapeLength), in] */__int64 * shape,
                        /* [in] */unsigned int dataLength,
                        /* [size_is(dataLength), in] */BYTE * data,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorInt8Bit * * result
                        ) = 0;
                    virtual HRESULT STDMETHODCALLTYPE CreateFromBuffer(
                        /* [in] */unsigned int shapeLength,
                        /* [size_is(shapeLength), in] */__int64 * shape,
                        /* [in] */ABI::Windows::Storage::Streams::IBuffer * buffer,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorInt8Bit * * result
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ITensorInt8BitStatics2=_uuidof(ITensorInt8BitStatics2);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics2;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics2_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000


/*
 *
 * Class Microsoft.AI.MachineLearning.TensorInt8Bit
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * RuntimeClass contains static methods.
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorInt8BitStatics2 interface starting with version 2.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorInt8BitStatics interface starting with version 1.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.ITensorInt8Bit ** Default Interface **
 *    Microsoft.AI.MachineLearning.ITensor
 *    Windows.Foundation.IMemoryBuffer
 *
 * Class Threading Model:  Both Single and Multi Threaded Apartment
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorInt8Bit_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorInt8Bit_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_TensorInt8Bit[] = L"Microsoft.AI.MachineLearning.TensorInt8Bit";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                class TensorUInt16Bit;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */



/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorUInt16Bit
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorUInt16Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorUInt16Bit:HRESULT GetAsVectorView(Windows.Foundation.Collections.IVectorView`1<UInt16>**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16Bit_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16Bit_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorUInt16Bit[] = L"Microsoft.AI.MachineLearning.ITensorUInt16Bit";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("33ea36fb-5129-5c13-b591-f407a6bfe637"), contract, object, exclusiveto] */
                MIDL_INTERFACE("33ea36fb-5129-5c13-b591-f407a6bfe637")
                ITensorUInt16Bit : public IInspectable
                {
                public:
                    virtual HRESULT STDMETHODCALLTYPE GetAsVectorView(
                        /* [out, retval] */__FIVectorView_1_UINT16 * * result
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ITensorUInt16Bit=_uuidof(ITensorUInt16Bit);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16Bit;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16Bit_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorUInt16BitStatics
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorUInt16Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorUInt16BitStatics:HRESULT Create(Microsoft.AI.MachineLearning.TensorUInt16Bit**);HRESULT Create2(Windows.Foundation.Collections.IIterable`1<Int64>*,Microsoft.AI.MachineLearning.TensorUInt16Bit**);HRESULT CreateFromArray(Windows.Foundation.Collections.IIterable`1<Int64>*,UInt32,UInt16*,Microsoft.AI.MachineLearning.TensorUInt16Bit**);HRESULT CreateFromIterable(Windows.Foundation.Collections.IIterable`1<Int64>*,Windows.Foundation.Collections.IIterable`1<UInt16>*,Microsoft.AI.MachineLearning.TensorUInt16Bit**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorUInt16BitStatics[] = L"Microsoft.AI.MachineLearning.ITensorUInt16BitStatics";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("93b94404-8dfc-592b-b51b-fbcf79261401"), contract, object, exclusiveto] */
                MIDL_INTERFACE("93b94404-8dfc-592b-b51b-fbcf79261401")
                ITensorUInt16BitStatics : public IInspectable
                {
                public:
                    /* [overload] */virtual HRESULT STDMETHODCALLTYPE Create(
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorUInt16Bit * * result
                        ) = 0;
                    /* [overload] */virtual HRESULT STDMETHODCALLTYPE Create2(
                        /* [in] */__FIIterable_1___z__zint64 * shape,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorUInt16Bit * * result
                        ) = 0;
                    virtual HRESULT STDMETHODCALLTYPE CreateFromArray(
                        /* [in] */__FIIterable_1___z__zint64 * shape,
                        /* [in] */unsigned int dataLength,
                        /* [size_is(dataLength), in] */unsigned short * data,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorUInt16Bit * * result
                        ) = 0;
                    virtual HRESULT STDMETHODCALLTYPE CreateFromIterable(
                        /* [in] */__FIIterable_1___z__zint64 * shape,
                        /* [in] */__FIIterable_1_UINT16 * data,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorUInt16Bit * * result
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ITensorUInt16BitStatics=_uuidof(ITensorUInt16BitStatics);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorUInt16BitStatics2
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 2.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorUInt16Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorUInt16BitStatics2:HRESULT CreateFromShapeArrayAndDataArray(UInt32,Int64*,UInt32,UInt16*,Microsoft.AI.MachineLearning.TensorUInt16Bit**);HRESULT CreateFromBuffer(UInt32,Int64*,Windows.Storage.Streams.IBuffer*,Microsoft.AI.MachineLearning.TensorUInt16Bit**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics2_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics2_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorUInt16BitStatics2[] = L"Microsoft.AI.MachineLearning.ITensorUInt16BitStatics2";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("8cc5af95-d59e-56de-aec9-c43cf9a7271f"), contract, object, exclusiveto] */
                MIDL_INTERFACE("8cc5af95-d59e-56de-aec9-c43cf9a7271f")
                ITensorUInt16BitStatics2 : public IInspectable
                {
                public:
                    virtual HRESULT STDMETHODCALLTYPE CreateFromShapeArrayAndDataArray(
                        /* [in] */unsigned int shapeLength,
                        /* [size_is(shapeLength), in] */__int64 * shape,
                        /* [in] */unsigned int dataLength,
                        /* [size_is(dataLength), in] */unsigned short * data,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorUInt16Bit * * result
                        ) = 0;
                    virtual HRESULT STDMETHODCALLTYPE CreateFromBuffer(
                        /* [in] */unsigned int shapeLength,
                        /* [size_is(shapeLength), in] */__int64 * shape,
                        /* [in] */ABI::Windows::Storage::Streams::IBuffer * buffer,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorUInt16Bit * * result
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ITensorUInt16BitStatics2=_uuidof(ITensorUInt16BitStatics2);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics2;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics2_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000


/*
 *
 * Class Microsoft.AI.MachineLearning.TensorUInt16Bit
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * RuntimeClass contains static methods.
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorUInt16BitStatics2 interface starting with version 2.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorUInt16BitStatics interface starting with version 1.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.ITensorUInt16Bit ** Default Interface **
 *    Microsoft.AI.MachineLearning.ITensor
 *    Windows.Foundation.IMemoryBuffer
 *
 * Class Threading Model:  Both Single and Multi Threaded Apartment
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorUInt16Bit_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorUInt16Bit_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_TensorUInt16Bit[] = L"Microsoft.AI.MachineLearning.TensorUInt16Bit";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                class TensorInt16Bit;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */



/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorInt16Bit
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorInt16Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorInt16Bit:HRESULT GetAsVectorView(Windows.Foundation.Collections.IVectorView`1<Int16>**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16Bit_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16Bit_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorInt16Bit[] = L"Microsoft.AI.MachineLearning.ITensorInt16Bit";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("a757726f-ddf3-5d6f-ab2e-17e5aad63088"), contract, object, exclusiveto] */
                MIDL_INTERFACE("a757726f-ddf3-5d6f-ab2e-17e5aad63088")
                ITensorInt16Bit : public IInspectable
                {
                public:
                    virtual HRESULT STDMETHODCALLTYPE GetAsVectorView(
                        /* [out, retval] */__FIVectorView_1_short * * result
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ITensorInt16Bit=_uuidof(ITensorInt16Bit);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16Bit;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16Bit_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorInt16BitStatics
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorInt16Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorInt16BitStatics:HRESULT Create(Microsoft.AI.MachineLearning.TensorInt16Bit**);HRESULT Create2(Windows.Foundation.Collections.IIterable`1<Int64>*,Microsoft.AI.MachineLearning.TensorInt16Bit**);HRESULT CreateFromArray(Windows.Foundation.Collections.IIterable`1<Int64>*,UInt32,Int16*,Microsoft.AI.MachineLearning.TensorInt16Bit**);HRESULT CreateFromIterable(Windows.Foundation.Collections.IIterable`1<Int64>*,Windows.Foundation.Collections.IIterable`1<Int16>*,Microsoft.AI.MachineLearning.TensorInt16Bit**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorInt16BitStatics[] = L"Microsoft.AI.MachineLearning.ITensorInt16BitStatics";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("5cabb6d2-4ee9-5967-ae6f-59d4a1c6d846"), contract, object, exclusiveto] */
                MIDL_INTERFACE("5cabb6d2-4ee9-5967-ae6f-59d4a1c6d846")
                ITensorInt16BitStatics : public IInspectable
                {
                public:
                    /* [overload] */virtual HRESULT STDMETHODCALLTYPE Create(
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorInt16Bit * * result
                        ) = 0;
                    /* [overload] */virtual HRESULT STDMETHODCALLTYPE Create2(
                        /* [in] */__FIIterable_1___z__zint64 * shape,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorInt16Bit * * result
                        ) = 0;
                    virtual HRESULT STDMETHODCALLTYPE CreateFromArray(
                        /* [in] */__FIIterable_1___z__zint64 * shape,
                        /* [in] */unsigned int dataLength,
                        /* [size_is(dataLength), in] */short * data,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorInt16Bit * * result
                        ) = 0;
                    virtual HRESULT STDMETHODCALLTYPE CreateFromIterable(
                        /* [in] */__FIIterable_1___z__zint64 * shape,
                        /* [in] */__FIIterable_1_short * data,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorInt16Bit * * result
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ITensorInt16BitStatics=_uuidof(ITensorInt16BitStatics);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorInt16BitStatics2
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 2.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorInt16Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorInt16BitStatics2:HRESULT CreateFromShapeArrayAndDataArray(UInt32,Int64*,UInt32,Int16*,Microsoft.AI.MachineLearning.TensorInt16Bit**);HRESULT CreateFromBuffer(UInt32,Int64*,Windows.Storage.Streams.IBuffer*,Microsoft.AI.MachineLearning.TensorInt16Bit**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics2_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics2_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorInt16BitStatics2[] = L"Microsoft.AI.MachineLearning.ITensorInt16BitStatics2";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("91eee292-5dbc-542f-a3d4-c675e6b7c31d"), contract, object, exclusiveto] */
                MIDL_INTERFACE("91eee292-5dbc-542f-a3d4-c675e6b7c31d")
                ITensorInt16BitStatics2 : public IInspectable
                {
                public:
                    virtual HRESULT STDMETHODCALLTYPE CreateFromShapeArrayAndDataArray(
                        /* [in] */unsigned int shapeLength,
                        /* [size_is(shapeLength), in] */__int64 * shape,
                        /* [in] */unsigned int dataLength,
                        /* [size_is(dataLength), in] */short * data,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorInt16Bit * * result
                        ) = 0;
                    virtual HRESULT STDMETHODCALLTYPE CreateFromBuffer(
                        /* [in] */unsigned int shapeLength,
                        /* [size_is(shapeLength), in] */__int64 * shape,
                        /* [in] */ABI::Windows::Storage::Streams::IBuffer * buffer,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorInt16Bit * * result
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ITensorInt16BitStatics2=_uuidof(ITensorInt16BitStatics2);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics2;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics2_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000


/*
 *
 * Class Microsoft.AI.MachineLearning.TensorInt16Bit
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * RuntimeClass contains static methods.
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorInt16BitStatics2 interface starting with version 2.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorInt16BitStatics interface starting with version 1.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.ITensorInt16Bit ** Default Interface **
 *    Microsoft.AI.MachineLearning.ITensor
 *    Windows.Foundation.IMemoryBuffer
 *
 * Class Threading Model:  Both Single and Multi Threaded Apartment
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorInt16Bit_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorInt16Bit_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_TensorInt16Bit[] = L"Microsoft.AI.MachineLearning.TensorInt16Bit";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                class TensorUInt32Bit;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */



/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorUInt32Bit
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorUInt32Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorUInt32Bit:HRESULT GetAsVectorView(Windows.Foundation.Collections.IVectorView`1<UInt32>**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32Bit_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32Bit_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorUInt32Bit[] = L"Microsoft.AI.MachineLearning.ITensorUInt32Bit";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("4453ec13-4a94-5d72-bd57-ac306c74d620"), contract, object, exclusiveto] */
                MIDL_INTERFACE("4453ec13-4a94-5d72-bd57-ac306c74d620")
                ITensorUInt32Bit : public IInspectable
                {
                public:
                    virtual HRESULT STDMETHODCALLTYPE GetAsVectorView(
                        /* [out, retval] */__FIVectorView_1_UINT32 * * result
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ITensorUInt32Bit=_uuidof(ITensorUInt32Bit);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32Bit;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32Bit_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorUInt32BitStatics
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorUInt32Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorUInt32BitStatics:HRESULT Create(Microsoft.AI.MachineLearning.TensorUInt32Bit**);HRESULT Create2(Windows.Foundation.Collections.IIterable`1<Int64>*,Microsoft.AI.MachineLearning.TensorUInt32Bit**);HRESULT CreateFromArray(Windows.Foundation.Collections.IIterable`1<Int64>*,UInt32,UInt32*,Microsoft.AI.MachineLearning.TensorUInt32Bit**);HRESULT CreateFromIterable(Windows.Foundation.Collections.IIterable`1<Int64>*,Windows.Foundation.Collections.IIterable`1<UInt32>*,Microsoft.AI.MachineLearning.TensorUInt32Bit**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorUInt32BitStatics[] = L"Microsoft.AI.MachineLearning.ITensorUInt32BitStatics";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("8b23c75c-3a66-5336-b509-45350f26bf09"), contract, object, exclusiveto] */
                MIDL_INTERFACE("8b23c75c-3a66-5336-b509-45350f26bf09")
                ITensorUInt32BitStatics : public IInspectable
                {
                public:
                    /* [overload] */virtual HRESULT STDMETHODCALLTYPE Create(
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorUInt32Bit * * result
                        ) = 0;
                    /* [overload] */virtual HRESULT STDMETHODCALLTYPE Create2(
                        /* [in] */__FIIterable_1___z__zint64 * shape,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorUInt32Bit * * result
                        ) = 0;
                    virtual HRESULT STDMETHODCALLTYPE CreateFromArray(
                        /* [in] */__FIIterable_1___z__zint64 * shape,
                        /* [in] */unsigned int dataLength,
                        /* [size_is(dataLength), in] */unsigned int * data,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorUInt32Bit * * result
                        ) = 0;
                    virtual HRESULT STDMETHODCALLTYPE CreateFromIterable(
                        /* [in] */__FIIterable_1___z__zint64 * shape,
                        /* [in] */__FIIterable_1_UINT32 * data,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorUInt32Bit * * result
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ITensorUInt32BitStatics=_uuidof(ITensorUInt32BitStatics);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorUInt32BitStatics2
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 2.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorUInt32Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorUInt32BitStatics2:HRESULT CreateFromShapeArrayAndDataArray(UInt32,Int64*,UInt32,UInt32*,Microsoft.AI.MachineLearning.TensorUInt32Bit**);HRESULT CreateFromBuffer(UInt32,Int64*,Windows.Storage.Streams.IBuffer*,Microsoft.AI.MachineLearning.TensorUInt32Bit**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics2_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics2_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorUInt32BitStatics2[] = L"Microsoft.AI.MachineLearning.ITensorUInt32BitStatics2";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("b53b05b9-c812-5da6-bfa8-aaff5bfdae79"), contract, object, exclusiveto] */
                MIDL_INTERFACE("b53b05b9-c812-5da6-bfa8-aaff5bfdae79")
                ITensorUInt32BitStatics2 : public IInspectable
                {
                public:
                    virtual HRESULT STDMETHODCALLTYPE CreateFromShapeArrayAndDataArray(
                        /* [in] */unsigned int shapeLength,
                        /* [size_is(shapeLength), in] */__int64 * shape,
                        /* [in] */unsigned int dataLength,
                        /* [size_is(dataLength), in] */unsigned int * data,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorUInt32Bit * * result
                        ) = 0;
                    virtual HRESULT STDMETHODCALLTYPE CreateFromBuffer(
                        /* [in] */unsigned int shapeLength,
                        /* [size_is(shapeLength), in] */__int64 * shape,
                        /* [in] */ABI::Windows::Storage::Streams::IBuffer * buffer,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorUInt32Bit * * result
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ITensorUInt32BitStatics2=_uuidof(ITensorUInt32BitStatics2);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics2;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics2_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000


/*
 *
 * Class Microsoft.AI.MachineLearning.TensorUInt32Bit
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * RuntimeClass contains static methods.
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorUInt32BitStatics2 interface starting with version 2.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorUInt32BitStatics interface starting with version 1.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.ITensorUInt32Bit ** Default Interface **
 *    Microsoft.AI.MachineLearning.ITensor
 *    Windows.Foundation.IMemoryBuffer
 *
 * Class Threading Model:  Both Single and Multi Threaded Apartment
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorUInt32Bit_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorUInt32Bit_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_TensorUInt32Bit[] = L"Microsoft.AI.MachineLearning.TensorUInt32Bit";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                class TensorInt32Bit;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */



/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorInt32Bit
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorInt32Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorInt32Bit:HRESULT GetAsVectorView(Windows.Foundation.Collections.IVectorView`1<Int32>**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32Bit_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32Bit_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorInt32Bit[] = L"Microsoft.AI.MachineLearning.ITensorInt32Bit";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("0daf5ea0-661f-5bc6-b210-2960138c688e"), contract, object, exclusiveto] */
                MIDL_INTERFACE("0daf5ea0-661f-5bc6-b210-2960138c688e")
                ITensorInt32Bit : public IInspectable
                {
                public:
                    virtual HRESULT STDMETHODCALLTYPE GetAsVectorView(
                        /* [out, retval] */__FIVectorView_1_int * * result
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ITensorInt32Bit=_uuidof(ITensorInt32Bit);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32Bit;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32Bit_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorInt32BitStatics
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorInt32Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorInt32BitStatics:HRESULT Create(Microsoft.AI.MachineLearning.TensorInt32Bit**);HRESULT Create2(Windows.Foundation.Collections.IIterable`1<Int64>*,Microsoft.AI.MachineLearning.TensorInt32Bit**);HRESULT CreateFromArray(Windows.Foundation.Collections.IIterable`1<Int64>*,UInt32,Int32*,Microsoft.AI.MachineLearning.TensorInt32Bit**);HRESULT CreateFromIterable(Windows.Foundation.Collections.IIterable`1<Int64>*,Windows.Foundation.Collections.IIterable`1<Int32>*,Microsoft.AI.MachineLearning.TensorInt32Bit**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorInt32BitStatics[] = L"Microsoft.AI.MachineLearning.ITensorInt32BitStatics";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("de454982-a86e-54aa-b88d-e34b66b3c001"), contract, object, exclusiveto] */
                MIDL_INTERFACE("de454982-a86e-54aa-b88d-e34b66b3c001")
                ITensorInt32BitStatics : public IInspectable
                {
                public:
                    /* [overload] */virtual HRESULT STDMETHODCALLTYPE Create(
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorInt32Bit * * result
                        ) = 0;
                    /* [overload] */virtual HRESULT STDMETHODCALLTYPE Create2(
                        /* [in] */__FIIterable_1___z__zint64 * shape,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorInt32Bit * * result
                        ) = 0;
                    virtual HRESULT STDMETHODCALLTYPE CreateFromArray(
                        /* [in] */__FIIterable_1___z__zint64 * shape,
                        /* [in] */unsigned int dataLength,
                        /* [size_is(dataLength), in] */int * data,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorInt32Bit * * result
                        ) = 0;
                    virtual HRESULT STDMETHODCALLTYPE CreateFromIterable(
                        /* [in] */__FIIterable_1___z__zint64 * shape,
                        /* [in] */__FIIterable_1_int * data,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorInt32Bit * * result
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ITensorInt32BitStatics=_uuidof(ITensorInt32BitStatics);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorInt32BitStatics2
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 2.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorInt32Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorInt32BitStatics2:HRESULT CreateFromShapeArrayAndDataArray(UInt32,Int64*,UInt32,Int32*,Microsoft.AI.MachineLearning.TensorInt32Bit**);HRESULT CreateFromBuffer(UInt32,Int64*,Windows.Storage.Streams.IBuffer*,Microsoft.AI.MachineLearning.TensorInt32Bit**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics2_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics2_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorInt32BitStatics2[] = L"Microsoft.AI.MachineLearning.ITensorInt32BitStatics2";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("a7924918-4afe-51d4-9f84-5dee7bdbe500"), contract, object, exclusiveto] */
                MIDL_INTERFACE("a7924918-4afe-51d4-9f84-5dee7bdbe500")
                ITensorInt32BitStatics2 : public IInspectable
                {
                public:
                    virtual HRESULT STDMETHODCALLTYPE CreateFromShapeArrayAndDataArray(
                        /* [in] */unsigned int shapeLength,
                        /* [size_is(shapeLength), in] */__int64 * shape,
                        /* [in] */unsigned int dataLength,
                        /* [size_is(dataLength), in] */int * data,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorInt32Bit * * result
                        ) = 0;
                    virtual HRESULT STDMETHODCALLTYPE CreateFromBuffer(
                        /* [in] */unsigned int shapeLength,
                        /* [size_is(shapeLength), in] */__int64 * shape,
                        /* [in] */ABI::Windows::Storage::Streams::IBuffer * buffer,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorInt32Bit * * result
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ITensorInt32BitStatics2=_uuidof(ITensorInt32BitStatics2);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics2;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics2_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000


/*
 *
 * Class Microsoft.AI.MachineLearning.TensorInt32Bit
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * RuntimeClass contains static methods.
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorInt32BitStatics2 interface starting with version 2.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorInt32BitStatics interface starting with version 1.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.ITensorInt32Bit ** Default Interface **
 *    Microsoft.AI.MachineLearning.ITensor
 *    Windows.Foundation.IMemoryBuffer
 *
 * Class Threading Model:  Both Single and Multi Threaded Apartment
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorInt32Bit_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorInt32Bit_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_TensorInt32Bit[] = L"Microsoft.AI.MachineLearning.TensorInt32Bit";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                class TensorUInt64Bit;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */



/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorUInt64Bit
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorUInt64Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorUInt64Bit:HRESULT GetAsVectorView(Windows.Foundation.Collections.IVectorView`1<UInt64>**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64Bit_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64Bit_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorUInt64Bit[] = L"Microsoft.AI.MachineLearning.ITensorUInt64Bit";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("c04b9e9b-fc04-58f4-b39c-c04d7f5777ea"), contract, object, exclusiveto] */
                MIDL_INTERFACE("c04b9e9b-fc04-58f4-b39c-c04d7f5777ea")
                ITensorUInt64Bit : public IInspectable
                {
                public:
                    virtual HRESULT STDMETHODCALLTYPE GetAsVectorView(
                        /* [out, retval] */__FIVectorView_1_UINT64 * * result
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ITensorUInt64Bit=_uuidof(ITensorUInt64Bit);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64Bit;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64Bit_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorUInt64BitStatics
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorUInt64Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorUInt64BitStatics:HRESULT Create(Microsoft.AI.MachineLearning.TensorUInt64Bit**);HRESULT Create2(Windows.Foundation.Collections.IIterable`1<Int64>*,Microsoft.AI.MachineLearning.TensorUInt64Bit**);HRESULT CreateFromArray(Windows.Foundation.Collections.IIterable`1<Int64>*,UInt32,UInt64*,Microsoft.AI.MachineLearning.TensorUInt64Bit**);HRESULT CreateFromIterable(Windows.Foundation.Collections.IIterable`1<Int64>*,Windows.Foundation.Collections.IIterable`1<UInt64>*,Microsoft.AI.MachineLearning.TensorUInt64Bit**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorUInt64BitStatics[] = L"Microsoft.AI.MachineLearning.ITensorUInt64BitStatics";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("0b04cac1-75b4-5e7f-b211-a56ddb3b5f47"), contract, object, exclusiveto] */
                MIDL_INTERFACE("0b04cac1-75b4-5e7f-b211-a56ddb3b5f47")
                ITensorUInt64BitStatics : public IInspectable
                {
                public:
                    /* [overload] */virtual HRESULT STDMETHODCALLTYPE Create(
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorUInt64Bit * * result
                        ) = 0;
                    /* [overload] */virtual HRESULT STDMETHODCALLTYPE Create2(
                        /* [in] */__FIIterable_1___z__zint64 * shape,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorUInt64Bit * * result
                        ) = 0;
                    virtual HRESULT STDMETHODCALLTYPE CreateFromArray(
                        /* [in] */__FIIterable_1___z__zint64 * shape,
                        /* [in] */unsigned int dataLength,
                        /* [size_is(dataLength), in] */unsigned __int64 * data,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorUInt64Bit * * result
                        ) = 0;
                    virtual HRESULT STDMETHODCALLTYPE CreateFromIterable(
                        /* [in] */__FIIterable_1___z__zint64 * shape,
                        /* [in] */__FIIterable_1_UINT64 * data,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorUInt64Bit * * result
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ITensorUInt64BitStatics=_uuidof(ITensorUInt64BitStatics);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorUInt64BitStatics2
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 2.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorUInt64Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorUInt64BitStatics2:HRESULT CreateFromShapeArrayAndDataArray(UInt32,Int64*,UInt32,UInt64*,Microsoft.AI.MachineLearning.TensorUInt64Bit**);HRESULT CreateFromBuffer(UInt32,Int64*,Windows.Storage.Streams.IBuffer*,Microsoft.AI.MachineLearning.TensorUInt64Bit**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics2_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics2_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorUInt64BitStatics2[] = L"Microsoft.AI.MachineLearning.ITensorUInt64BitStatics2";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("d3e5a1ed-c9e7-555a-8b9f-a0773b447ae8"), contract, object, exclusiveto] */
                MIDL_INTERFACE("d3e5a1ed-c9e7-555a-8b9f-a0773b447ae8")
                ITensorUInt64BitStatics2 : public IInspectable
                {
                public:
                    virtual HRESULT STDMETHODCALLTYPE CreateFromShapeArrayAndDataArray(
                        /* [in] */unsigned int shapeLength,
                        /* [size_is(shapeLength), in] */__int64 * shape,
                        /* [in] */unsigned int dataLength,
                        /* [size_is(dataLength), in] */unsigned __int64 * data,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorUInt64Bit * * result
                        ) = 0;
                    virtual HRESULT STDMETHODCALLTYPE CreateFromBuffer(
                        /* [in] */unsigned int shapeLength,
                        /* [size_is(shapeLength), in] */__int64 * shape,
                        /* [in] */ABI::Windows::Storage::Streams::IBuffer * buffer,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorUInt64Bit * * result
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ITensorUInt64BitStatics2=_uuidof(ITensorUInt64BitStatics2);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics2;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics2_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000


/*
 *
 * Class Microsoft.AI.MachineLearning.TensorUInt64Bit
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * RuntimeClass contains static methods.
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorUInt64BitStatics2 interface starting with version 2.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorUInt64BitStatics interface starting with version 1.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.ITensorUInt64Bit ** Default Interface **
 *    Microsoft.AI.MachineLearning.ITensor
 *    Windows.Foundation.IMemoryBuffer
 *
 * Class Threading Model:  Both Single and Multi Threaded Apartment
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorUInt64Bit_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorUInt64Bit_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_TensorUInt64Bit[] = L"Microsoft.AI.MachineLearning.TensorUInt64Bit";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                class TensorInt64Bit;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */



/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorInt64Bit
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorInt64Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorInt64Bit:HRESULT GetAsVectorView(Windows.Foundation.Collections.IVectorView`1<Int64>**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64Bit_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64Bit_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorInt64Bit[] = L"Microsoft.AI.MachineLearning.ITensorInt64Bit";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("330157f0-a3c0-5b99-b593-3bac3103e9b3"), contract, object, exclusiveto] */
                MIDL_INTERFACE("330157f0-a3c0-5b99-b593-3bac3103e9b3")
                ITensorInt64Bit : public IInspectable
                {
                public:
                    virtual HRESULT STDMETHODCALLTYPE GetAsVectorView(
                        /* [out, retval] */__FIVectorView_1___z__zint64 * * result
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ITensorInt64Bit=_uuidof(ITensorInt64Bit);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64Bit;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64Bit_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorInt64BitStatics
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorInt64Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorInt64BitStatics:HRESULT Create(Microsoft.AI.MachineLearning.TensorInt64Bit**);HRESULT Create2(Windows.Foundation.Collections.IIterable`1<Int64>*,Microsoft.AI.MachineLearning.TensorInt64Bit**);HRESULT CreateFromArray(Windows.Foundation.Collections.IIterable`1<Int64>*,UInt32,Int64*,Microsoft.AI.MachineLearning.TensorInt64Bit**);HRESULT CreateFromIterable(Windows.Foundation.Collections.IIterable`1<Int64>*,Windows.Foundation.Collections.IIterable`1<Int64>*,Microsoft.AI.MachineLearning.TensorInt64Bit**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorInt64BitStatics[] = L"Microsoft.AI.MachineLearning.ITensorInt64BitStatics";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("d3aff765-eb3c-5e0f-959b-16c66ecfa5f2"), contract, object, exclusiveto] */
                MIDL_INTERFACE("d3aff765-eb3c-5e0f-959b-16c66ecfa5f2")
                ITensorInt64BitStatics : public IInspectable
                {
                public:
                    /* [overload] */virtual HRESULT STDMETHODCALLTYPE Create(
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorInt64Bit * * result
                        ) = 0;
                    /* [overload] */virtual HRESULT STDMETHODCALLTYPE Create2(
                        /* [in] */__FIIterable_1___z__zint64 * shape,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorInt64Bit * * result
                        ) = 0;
                    virtual HRESULT STDMETHODCALLTYPE CreateFromArray(
                        /* [in] */__FIIterable_1___z__zint64 * shape,
                        /* [in] */unsigned int dataLength,
                        /* [size_is(dataLength), in] */__int64 * data,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorInt64Bit * * result
                        ) = 0;
                    virtual HRESULT STDMETHODCALLTYPE CreateFromIterable(
                        /* [in] */__FIIterable_1___z__zint64 * shape,
                        /* [in] */__FIIterable_1___z__zint64 * data,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorInt64Bit * * result
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ITensorInt64BitStatics=_uuidof(ITensorInt64BitStatics);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorInt64BitStatics2
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 2.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorInt64Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorInt64BitStatics2:HRESULT CreateFromShapeArrayAndDataArray(UInt32,Int64*,UInt32,Int64*,Microsoft.AI.MachineLearning.TensorInt64Bit**);HRESULT CreateFromBuffer(UInt32,Int64*,Windows.Storage.Streams.IBuffer*,Microsoft.AI.MachineLearning.TensorInt64Bit**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics2_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics2_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorInt64BitStatics2[] = L"Microsoft.AI.MachineLearning.ITensorInt64BitStatics2";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("bf724240-4b4b-5434-9a1a-5f274ec542f8"), contract, object, exclusiveto] */
                MIDL_INTERFACE("bf724240-4b4b-5434-9a1a-5f274ec542f8")
                ITensorInt64BitStatics2 : public IInspectable
                {
                public:
                    virtual HRESULT STDMETHODCALLTYPE CreateFromShapeArrayAndDataArray(
                        /* [in] */unsigned int shapeLength,
                        /* [size_is(shapeLength), in] */__int64 * shape,
                        /* [in] */unsigned int dataLength,
                        /* [size_is(dataLength), in] */__int64 * data,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorInt64Bit * * result
                        ) = 0;
                    virtual HRESULT STDMETHODCALLTYPE CreateFromBuffer(
                        /* [in] */unsigned int shapeLength,
                        /* [size_is(shapeLength), in] */__int64 * shape,
                        /* [in] */ABI::Windows::Storage::Streams::IBuffer * buffer,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorInt64Bit * * result
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ITensorInt64BitStatics2=_uuidof(ITensorInt64BitStatics2);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics2;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics2_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000


/*
 *
 * Class Microsoft.AI.MachineLearning.TensorInt64Bit
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * RuntimeClass contains static methods.
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorInt64BitStatics2 interface starting with version 2.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorInt64BitStatics interface starting with version 1.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.ITensorInt64Bit ** Default Interface **
 *    Microsoft.AI.MachineLearning.ITensor
 *    Windows.Foundation.IMemoryBuffer
 *
 * Class Threading Model:  Both Single and Multi Threaded Apartment
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorInt64Bit_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorInt64Bit_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_TensorInt64Bit[] = L"Microsoft.AI.MachineLearning.TensorInt64Bit";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                class TensorBoolean;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */



/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorBoolean
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorBoolean
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorBoolean:HRESULT GetAsVectorView(Windows.Foundation.Collections.IVectorView`1<Boolean>**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBoolean_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBoolean_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorBoolean[] = L"Microsoft.AI.MachineLearning.ITensorBoolean";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("5e234751-afc1-547b-a05f-3be1401d7f6b"), contract, object, exclusiveto] */
                MIDL_INTERFACE("5e234751-afc1-547b-a05f-3be1401d7f6b")
                ITensorBoolean : public IInspectable
                {
                public:
                    virtual HRESULT STDMETHODCALLTYPE GetAsVectorView(
                        /* [out, retval] */__FIVectorView_1_boolean * * result
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ITensorBoolean=_uuidof(ITensorBoolean);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBoolean;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBoolean_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorBooleanStatics
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorBoolean
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorBooleanStatics:HRESULT Create(Microsoft.AI.MachineLearning.TensorBoolean**);HRESULT Create2(Windows.Foundation.Collections.IIterable`1<Int64>*,Microsoft.AI.MachineLearning.TensorBoolean**);HRESULT CreateFromArray(Windows.Foundation.Collections.IIterable`1<Int64>*,UInt32,Boolean*,Microsoft.AI.MachineLearning.TensorBoolean**);HRESULT CreateFromIterable(Windows.Foundation.Collections.IIterable`1<Int64>*,Windows.Foundation.Collections.IIterable`1<Boolean>*,Microsoft.AI.MachineLearning.TensorBoolean**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorBooleanStatics[] = L"Microsoft.AI.MachineLearning.ITensorBooleanStatics";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("d9f16dbc-e102-587d-bacb-f31c98da0bb3"), contract, object, exclusiveto] */
                MIDL_INTERFACE("d9f16dbc-e102-587d-bacb-f31c98da0bb3")
                ITensorBooleanStatics : public IInspectable
                {
                public:
                    /* [overload] */virtual HRESULT STDMETHODCALLTYPE Create(
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorBoolean * * result
                        ) = 0;
                    /* [overload] */virtual HRESULT STDMETHODCALLTYPE Create2(
                        /* [in] */__FIIterable_1___z__zint64 * shape,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorBoolean * * result
                        ) = 0;
                    virtual HRESULT STDMETHODCALLTYPE CreateFromArray(
                        /* [in] */__FIIterable_1___z__zint64 * shape,
                        /* [in] */unsigned int dataLength,
                        /* [size_is(dataLength), in] */::boolean * data,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorBoolean * * result
                        ) = 0;
                    virtual HRESULT STDMETHODCALLTYPE CreateFromIterable(
                        /* [in] */__FIIterable_1___z__zint64 * shape,
                        /* [in] */__FIIterable_1_boolean * data,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorBoolean * * result
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ITensorBooleanStatics=_uuidof(ITensorBooleanStatics);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorBooleanStatics2
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 2.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorBoolean
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorBooleanStatics2:HRESULT CreateFromShapeArrayAndDataArray(UInt32,Int64*,UInt32,Boolean*,Microsoft.AI.MachineLearning.TensorBoolean**);HRESULT CreateFromBuffer(UInt32,Int64*,Windows.Storage.Streams.IBuffer*,Microsoft.AI.MachineLearning.TensorBoolean**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics2_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics2_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorBooleanStatics2[] = L"Microsoft.AI.MachineLearning.ITensorBooleanStatics2";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("ff1310fb-472e-5a85-965b-6fc0502e575f"), contract, object, exclusiveto] */
                MIDL_INTERFACE("ff1310fb-472e-5a85-965b-6fc0502e575f")
                ITensorBooleanStatics2 : public IInspectable
                {
                public:
                    virtual HRESULT STDMETHODCALLTYPE CreateFromShapeArrayAndDataArray(
                        /* [in] */unsigned int shapeLength,
                        /* [size_is(shapeLength), in] */__int64 * shape,
                        /* [in] */unsigned int dataLength,
                        /* [size_is(dataLength), in] */::boolean * data,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorBoolean * * result
                        ) = 0;
                    virtual HRESULT STDMETHODCALLTYPE CreateFromBuffer(
                        /* [in] */unsigned int shapeLength,
                        /* [size_is(shapeLength), in] */__int64 * shape,
                        /* [in] */ABI::Windows::Storage::Streams::IBuffer * buffer,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorBoolean * * result
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ITensorBooleanStatics2=_uuidof(ITensorBooleanStatics2);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics2;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics2_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000


/*
 *
 * Class Microsoft.AI.MachineLearning.TensorBoolean
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * RuntimeClass contains static methods.
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorBooleanStatics2 interface starting with version 2.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorBooleanStatics interface starting with version 1.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.ITensorBoolean ** Default Interface **
 *    Microsoft.AI.MachineLearning.ITensor
 *    Windows.Foundation.IMemoryBuffer
 *
 * Class Threading Model:  Both Single and Multi Threaded Apartment
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorBoolean_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorBoolean_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_TensorBoolean[] = L"Microsoft.AI.MachineLearning.TensorBoolean";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                class TensorDouble;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */



/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorDouble
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorDouble
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorDouble:HRESULT GetAsVectorView(Windows.Foundation.Collections.IVectorView`1<Double>**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDouble_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDouble_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorDouble[] = L"Microsoft.AI.MachineLearning.ITensorDouble";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("742a3b98-b0ea-59b3-b0f5-f257608df0c3"), contract, object, exclusiveto] */
                MIDL_INTERFACE("742a3b98-b0ea-59b3-b0f5-f257608df0c3")
                ITensorDouble : public IInspectable
                {
                public:
                    virtual HRESULT STDMETHODCALLTYPE GetAsVectorView(
                        /* [out, retval] */__FIVectorView_1_double * * result
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ITensorDouble=_uuidof(ITensorDouble);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDouble;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDouble_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorDoubleStatics
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorDouble
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorDoubleStatics:HRESULT Create(Microsoft.AI.MachineLearning.TensorDouble**);HRESULT Create2(Windows.Foundation.Collections.IIterable`1<Int64>*,Microsoft.AI.MachineLearning.TensorDouble**);HRESULT CreateFromArray(Windows.Foundation.Collections.IIterable`1<Int64>*,UInt32,Double*,Microsoft.AI.MachineLearning.TensorDouble**);HRESULT CreateFromIterable(Windows.Foundation.Collections.IIterable`1<Int64>*,Windows.Foundation.Collections.IIterable`1<Double>*,Microsoft.AI.MachineLearning.TensorDouble**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorDoubleStatics[] = L"Microsoft.AI.MachineLearning.ITensorDoubleStatics";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("2c4821cf-0cb2-567b-b8d5-980bae99c2e8"), contract, object, exclusiveto] */
                MIDL_INTERFACE("2c4821cf-0cb2-567b-b8d5-980bae99c2e8")
                ITensorDoubleStatics : public IInspectable
                {
                public:
                    /* [overload] */virtual HRESULT STDMETHODCALLTYPE Create(
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorDouble * * result
                        ) = 0;
                    /* [overload] */virtual HRESULT STDMETHODCALLTYPE Create2(
                        /* [in] */__FIIterable_1___z__zint64 * shape,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorDouble * * result
                        ) = 0;
                    virtual HRESULT STDMETHODCALLTYPE CreateFromArray(
                        /* [in] */__FIIterable_1___z__zint64 * shape,
                        /* [in] */unsigned int dataLength,
                        /* [size_is(dataLength), in] */double * data,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorDouble * * result
                        ) = 0;
                    virtual HRESULT STDMETHODCALLTYPE CreateFromIterable(
                        /* [in] */__FIIterable_1___z__zint64 * shape,
                        /* [in] */__FIIterable_1_double * data,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorDouble * * result
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ITensorDoubleStatics=_uuidof(ITensorDoubleStatics);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorDoubleStatics2
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 2.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorDouble
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorDoubleStatics2:HRESULT CreateFromShapeArrayAndDataArray(UInt32,Int64*,UInt32,Double*,Microsoft.AI.MachineLearning.TensorDouble**);HRESULT CreateFromBuffer(UInt32,Int64*,Windows.Storage.Streams.IBuffer*,Microsoft.AI.MachineLearning.TensorDouble**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics2_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics2_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorDoubleStatics2[] = L"Microsoft.AI.MachineLearning.ITensorDoubleStatics2";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("1a4f5b44-280a-5947-97c2-4fc2fcdbc65b"), contract, object, exclusiveto] */
                MIDL_INTERFACE("1a4f5b44-280a-5947-97c2-4fc2fcdbc65b")
                ITensorDoubleStatics2 : public IInspectable
                {
                public:
                    virtual HRESULT STDMETHODCALLTYPE CreateFromShapeArrayAndDataArray(
                        /* [in] */unsigned int shapeLength,
                        /* [size_is(shapeLength), in] */__int64 * shape,
                        /* [in] */unsigned int dataLength,
                        /* [size_is(dataLength), in] */double * data,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorDouble * * result
                        ) = 0;
                    virtual HRESULT STDMETHODCALLTYPE CreateFromBuffer(
                        /* [in] */unsigned int shapeLength,
                        /* [size_is(shapeLength), in] */__int64 * shape,
                        /* [in] */ABI::Windows::Storage::Streams::IBuffer * buffer,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorDouble * * result
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ITensorDoubleStatics2=_uuidof(ITensorDoubleStatics2);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics2;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics2_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000


/*
 *
 * Class Microsoft.AI.MachineLearning.TensorDouble
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * RuntimeClass contains static methods.
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorDoubleStatics2 interface starting with version 2.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorDoubleStatics interface starting with version 1.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.ITensorDouble ** Default Interface **
 *    Microsoft.AI.MachineLearning.ITensor
 *    Windows.Foundation.IMemoryBuffer
 *
 * Class Threading Model:  Both Single and Multi Threaded Apartment
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorDouble_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorDouble_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_TensorDouble[] = L"Microsoft.AI.MachineLearning.TensorDouble";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                class TensorString;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */



/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorString
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorString
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorString:HRESULT GetAsVectorView(Windows.Foundation.Collections.IVectorView`1<String>**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorString_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorString_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorString[] = L"Microsoft.AI.MachineLearning.ITensorString";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("32e5741f-0457-5df4-aea4-812fc70b138f"), contract, object, exclusiveto] */
                MIDL_INTERFACE("32e5741f-0457-5df4-aea4-812fc70b138f")
                ITensorString : public IInspectable
                {
                public:
                    virtual HRESULT STDMETHODCALLTYPE GetAsVectorView(
                        /* [out, retval] */__FIVectorView_1_HSTRING * * result
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ITensorString=_uuidof(ITensorString);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorString;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorString_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorStringStatics
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorString
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorStringStatics:HRESULT Create(Microsoft.AI.MachineLearning.TensorString**);HRESULT Create2(Windows.Foundation.Collections.IIterable`1<Int64>*,Microsoft.AI.MachineLearning.TensorString**);HRESULT CreateFromArray(Windows.Foundation.Collections.IIterable`1<Int64>*,UInt32,String*,Microsoft.AI.MachineLearning.TensorString**);HRESULT CreateFromIterable(Windows.Foundation.Collections.IIterable`1<Int64>*,Windows.Foundation.Collections.IIterable`1<String>*,Microsoft.AI.MachineLearning.TensorString**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorStringStatics[] = L"Microsoft.AI.MachineLearning.ITensorStringStatics";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("a27ef7bd-b6d5-5d3d-bbbf-9012d76733e5"), contract, object, exclusiveto] */
                MIDL_INTERFACE("a27ef7bd-b6d5-5d3d-bbbf-9012d76733e5")
                ITensorStringStatics : public IInspectable
                {
                public:
                    /* [overload] */virtual HRESULT STDMETHODCALLTYPE Create(
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorString * * result
                        ) = 0;
                    /* [overload] */virtual HRESULT STDMETHODCALLTYPE Create2(
                        /* [in] */__FIIterable_1___z__zint64 * shape,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorString * * result
                        ) = 0;
                    virtual HRESULT STDMETHODCALLTYPE CreateFromArray(
                        /* [in] */__FIIterable_1___z__zint64 * shape,
                        /* [in] */unsigned int dataLength,
                        /* [size_is(dataLength), in] */HSTRING * data,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorString * * result
                        ) = 0;
                    virtual HRESULT STDMETHODCALLTYPE CreateFromIterable(
                        /* [in] */__FIIterable_1___z__zint64 * shape,
                        /* [in] */__FIIterable_1_HSTRING * data,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorString * * result
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ITensorStringStatics=_uuidof(ITensorStringStatics);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorStringStatics2
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 2.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorString
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorStringStatics2:HRESULT CreateFromShapeArrayAndDataArray(UInt32,Int64*,UInt32,String*,Microsoft.AI.MachineLearning.TensorString**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics2_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics2_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorStringStatics2[] = L"Microsoft.AI.MachineLearning.ITensorStringStatics2";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("1984cf8a-b839-5bc8-b2b5-865f7867edd4"), contract, object, exclusiveto] */
                MIDL_INTERFACE("1984cf8a-b839-5bc8-b2b5-865f7867edd4")
                ITensorStringStatics2 : public IInspectable
                {
                public:
                    virtual HRESULT STDMETHODCALLTYPE CreateFromShapeArrayAndDataArray(
                        /* [in] */unsigned int shapeLength,
                        /* [size_is(shapeLength), in] */__int64 * shape,
                        /* [in] */unsigned int dataLength,
                        /* [size_is(dataLength), in] */HSTRING * data,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::ITensorString * * result
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_ITensorStringStatics2=_uuidof(ITensorStringStatics2);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics2;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics2_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000


/*
 *
 * Class Microsoft.AI.MachineLearning.TensorString
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * RuntimeClass contains static methods.
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorStringStatics2 interface starting with version 2.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorStringStatics interface starting with version 1.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.ITensorString ** Default Interface **
 *    Microsoft.AI.MachineLearning.ITensor
 *    Windows.Foundation.IMemoryBuffer
 *
 * Class Threading Model:  Both Single and Multi Threaded Apartment
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorString_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorString_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_TensorString[] = L"Microsoft.AI.MachineLearning.TensorString";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                class ImageFeatureValue;
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */



/*
 *
 * Interface Microsoft.AI.MachineLearning.IImageFeatureValue
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.ImageFeatureValue
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.IImageFeatureValue:HRESULT get_VideoFrame(Windows.Media.VideoFrame**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValue_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValue_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_IImageFeatureValue[] = L"Microsoft.AI.MachineLearning.IImageFeatureValue";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("ebec66ba-18d9-5851-8bed-86559cbc3545"), contract, object, exclusiveto] */
                MIDL_INTERFACE("ebec66ba-18d9-5851-8bed-86559cbc3545")
                IImageFeatureValue : public IInspectable
                {
                public:
                    /* [propget] */virtual HRESULT STDMETHODCALLTYPE get_VideoFrame(
                        /* [out, retval] */ABI::Windows::Media::IVideoFrame * * value
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_IImageFeatureValue=_uuidof(IImageFeatureValue);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValue;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValue_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.IImageFeatureValueStatics
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.ImageFeatureValue
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.IImageFeatureValueStatics:HRESULT CreateFromVideoFrame(Windows.Media.VideoFrame*,Microsoft.AI.MachineLearning.ImageFeatureValue**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValueStatics_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValueStatics_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_IImageFeatureValueStatics[] = L"Microsoft.AI.MachineLearning.IImageFeatureValueStatics";
namespace ABI {
    namespace Microsoft {
        namespace AI {
            namespace MachineLearning {
                /* [uuid("c822d410-8e0e-5fec-9c61-3cb55cdd8590"), contract, object, exclusiveto] */
                MIDL_INTERFACE("c822d410-8e0e-5fec-9c61-3cb55cdd8590")
                IImageFeatureValueStatics : public IInspectable
                {
                public:
                    virtual HRESULT STDMETHODCALLTYPE CreateFromVideoFrame(
                        /* [in] */ABI::Windows::Media::IVideoFrame * image,
                        /* [out, retval] */ABI::Microsoft::AI::MachineLearning::IImageFeatureValue * * result
                        ) = 0;
                    
                };

                extern MIDL_CONST_ID IID & IID_IImageFeatureValueStatics=_uuidof(IImageFeatureValueStatics);
                
            } /* MachineLearning */
        } /* AI */
    } /* Microsoft */
} /* ABI */

EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValueStatics;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValueStatics_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Class Microsoft.AI.MachineLearning.ImageFeatureValue
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * RuntimeClass contains static methods.
 *   Static Methods exist on the Microsoft.AI.MachineLearning.IImageFeatureValueStatics interface starting with version 1.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.IImageFeatureValue ** Default Interface **
 *    Microsoft.AI.MachineLearning.ILearningModelFeatureValue
 *
 * Class Threading Model:  Both Single and Multi Threaded Apartment
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_ImageFeatureValue_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_ImageFeatureValue_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_ImageFeatureValue[] = L"Microsoft.AI.MachineLearning.ImageFeatureValue";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


#else // !defined(__cplusplus)
/* Forward Declarations */
#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelOperatorProvider_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelOperatorProvider_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelOperatorProvider __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelOperatorProvider;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelOperatorProvider_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDevice_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDevice_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDevice __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDevice;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDevice_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceStatics_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceStatics_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceStatics __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceStatics;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceStatics_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceFactory_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceFactory_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceFactory __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceFactory;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceFactory_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelEvaluationResult_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelEvaluationResult_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelEvaluationResult __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelEvaluationResult;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelEvaluationResult_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions2_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions2_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions2 __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions2;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions2_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions3_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions3_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions3 __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions3;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions3_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory2_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory2_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory2 __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory2;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory2_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureValue_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureValue_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureValue __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureValue;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureValue_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBinding_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBinding_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBinding __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBinding;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBinding_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBindingFactory_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBindingFactory_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBindingFactory __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBindingFactory;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBindingFactory_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CIMapFeatureDescriptor_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CIMapFeatureDescriptor_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CIMapFeatureDescriptor __x_ABI_CMicrosoft_CAI_CMachineLearning_CIMapFeatureDescriptor;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CIMapFeatureDescriptor_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CISequenceFeatureDescriptor_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CISequenceFeatureDescriptor_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CISequenceFeatureDescriptor __x_ABI_CMicrosoft_CAI_CMachineLearning_CISequenceFeatureDescriptor;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CISequenceFeatureDescriptor_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFeatureDescriptor_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFeatureDescriptor_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFeatureDescriptor __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFeatureDescriptor;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFeatureDescriptor_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor2_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor2_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor2 __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor2;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor2_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensor_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensor_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensor __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensor;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensor_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics2_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics2_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics2 __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics2;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics2_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16Bit_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16Bit_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16Bit __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16Bit;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16Bit_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics2_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics2_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics2 __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics2;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics2_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8Bit_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8Bit_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8Bit __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8Bit;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8Bit_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics2_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics2_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics2 __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics2;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics2_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8Bit_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8Bit_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8Bit __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8Bit;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8Bit_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics2_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics2_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics2 __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics2;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics2_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16Bit_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16Bit_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16Bit __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16Bit;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16Bit_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics2_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics2_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics2 __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics2;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics2_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16Bit_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16Bit_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16Bit __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16Bit;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16Bit_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics2_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics2_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics2 __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics2;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics2_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32Bit_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32Bit_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32Bit __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32Bit;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32Bit_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics2_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics2_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics2 __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics2;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics2_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32Bit_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32Bit_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32Bit __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32Bit;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32Bit_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics2_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics2_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics2 __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics2;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics2_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64Bit_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64Bit_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64Bit __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64Bit;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64Bit_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics2_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics2_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics2 __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics2;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics2_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64Bit_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64Bit_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64Bit __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64Bit;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64Bit_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics2_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics2_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics2 __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics2;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics2_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBoolean_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBoolean_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBoolean __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBoolean;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBoolean_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics2_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics2_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics2 __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics2;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics2_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDouble_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDouble_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDouble __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDouble;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDouble_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics2_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics2_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics2 __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics2;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics2_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorString_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorString_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorString __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorString;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorString_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics2_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics2_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics2 __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics2;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics2_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValue_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValue_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValue __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValue;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValue_FWD_DEFINED__

#ifndef ____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValueStatics_FWD_DEFINED__
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValueStatics_FWD_DEFINED__
typedef interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValueStatics __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValueStatics;

#endif // ____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValueStatics_FWD_DEFINED__

// Parameterized interface forward declarations (C)

// Collection interface definitions
#if !defined(____FIKeyValuePair_2_HSTRING_HSTRING_INTERFACE_DEFINED__)
#define ____FIKeyValuePair_2_HSTRING_HSTRING_INTERFACE_DEFINED__

typedef interface __FIKeyValuePair_2_HSTRING_HSTRING __FIKeyValuePair_2_HSTRING_HSTRING;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIKeyValuePair_2_HSTRING_HSTRING;

typedef struct __FIKeyValuePair_2_HSTRING_HSTRINGVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface )(__RPC__in __FIKeyValuePair_2_HSTRING_HSTRING * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);

    ULONG ( STDMETHODCALLTYPE *AddRef )(__RPC__in __FIKeyValuePair_2_HSTRING_HSTRING * This);
    ULONG ( STDMETHODCALLTYPE *Release )(__RPC__in __FIKeyValuePair_2_HSTRING_HSTRING * This);
    HRESULT ( STDMETHODCALLTYPE *GetIids )(__RPC__in __FIKeyValuePair_2_HSTRING_HSTRING * This,
            /* [out] */ __RPC__out ULONG *iidCount,
            /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);
    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(__RPC__in __FIKeyValuePair_2_HSTRING_HSTRING * This, /* [out] */ __RPC__deref_out_opt HSTRING *className);
    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(__RPC__in __FIKeyValuePair_2_HSTRING_HSTRING * This, /* [out] */ __RPC__out TrustLevel *trustLevel);

    /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Key )(__RPC__in __FIKeyValuePair_2_HSTRING_HSTRING * This, /* [retval][out] */ __RPC__out HSTRING *key);
    /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Value )(__RPC__in __FIKeyValuePair_2_HSTRING_HSTRING * This, /* [retval][out] */ __RPC__deref_out_opt HSTRING *value);
    END_INTERFACE
} __FIKeyValuePair_2_HSTRING_HSTRINGVtbl;

interface __FIKeyValuePair_2_HSTRING_HSTRING
{
    CONST_VTBL struct __FIKeyValuePair_2_HSTRING_HSTRINGVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __FIKeyValuePair_2_HSTRING_HSTRING_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define __FIKeyValuePair_2_HSTRING_HSTRING_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define __FIKeyValuePair_2_HSTRING_HSTRING_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define __FIKeyValuePair_2_HSTRING_HSTRING_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 

#define __FIKeyValuePair_2_HSTRING_HSTRING_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 

#define __FIKeyValuePair_2_HSTRING_HSTRING_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 


#define __FIKeyValuePair_2_HSTRING_HSTRING_get_Key(This,key)	\
    ( (This)->lpVtbl -> get_Key(This,key) ) 

#define __FIKeyValuePair_2_HSTRING_HSTRING_get_Value(This,value)	\
    ( (This)->lpVtbl -> get_Value(This,value) ) 
#endif /* COBJMACROS */


#endif // ____FIKeyValuePair_2_HSTRING_HSTRING_INTERFACE_DEFINED__



#if !defined(____FIIterator_1___FIKeyValuePair_2_HSTRING_HSTRING_INTERFACE_DEFINED__)
#define ____FIIterator_1___FIKeyValuePair_2_HSTRING_HSTRING_INTERFACE_DEFINED__

typedef interface __FIIterator_1___FIKeyValuePair_2_HSTRING_HSTRING __FIIterator_1___FIKeyValuePair_2_HSTRING_HSTRING;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIIterator_1___FIKeyValuePair_2_HSTRING_HSTRING;

typedef struct __FIIterator_1___FIKeyValuePair_2_HSTRING_HSTRINGVtbl
{
    BEGIN_INTERFACE

    HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
        __RPC__in __FIIterator_1___FIKeyValuePair_2_HSTRING_HSTRING * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);
    ULONG ( STDMETHODCALLTYPE *AddRef )(__RPC__in __FIIterator_1___FIKeyValuePair_2_HSTRING_HSTRING * This);
    ULONG ( STDMETHODCALLTYPE *Release )(__RPC__in __FIIterator_1___FIKeyValuePair_2_HSTRING_HSTRING * This);
    HRESULT ( STDMETHODCALLTYPE *GetIids )(__RPC__in __FIIterator_1___FIKeyValuePair_2_HSTRING_HSTRING * This,
        /* [out] */ __RPC__out ULONG *iidCount,
        /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);

    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(__RPC__in __FIIterator_1___FIKeyValuePair_2_HSTRING_HSTRING * This, /* [out] */ __RPC__deref_out_opt HSTRING *className);
    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(__RPC__in __FIIterator_1___FIKeyValuePair_2_HSTRING_HSTRING * This, /* [out] */ __RPC__out TrustLevel *trustLevel);

    /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Current )(__RPC__in __FIIterator_1___FIKeyValuePair_2_HSTRING_HSTRING * This, /* [retval][out] */ __RPC__out __FIKeyValuePair_2_HSTRING_HSTRING * *current);
    /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_HasCurrent )(__RPC__in __FIIterator_1___FIKeyValuePair_2_HSTRING_HSTRING * This, /* [retval][out] */ __RPC__out boolean *hasCurrent);
    HRESULT ( STDMETHODCALLTYPE *MoveNext )(__RPC__in __FIIterator_1___FIKeyValuePair_2_HSTRING_HSTRING * This, /* [retval][out] */ __RPC__out boolean *hasCurrent);
    HRESULT ( STDMETHODCALLTYPE *GetMany )(__RPC__in __FIIterator_1___FIKeyValuePair_2_HSTRING_HSTRING * This,
        /* [in] */ unsigned int capacity,
        /* [size_is][length_is][out] */ __RPC__out_ecount_part(capacity, *actual) __FIKeyValuePair_2_HSTRING_HSTRING * *items,
        /* [retval][out] */ __RPC__out unsigned int *actual);

    END_INTERFACE
} __FIIterator_1___FIKeyValuePair_2_HSTRING_HSTRINGVtbl;

interface __FIIterator_1___FIKeyValuePair_2_HSTRING_HSTRING
{
    CONST_VTBL struct __FIIterator_1___FIKeyValuePair_2_HSTRING_HSTRINGVtbl *lpVtbl;
};



#ifdef COBJMACROS


#define __FIIterator_1___FIKeyValuePair_2_HSTRING_HSTRING_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define __FIIterator_1___FIKeyValuePair_2_HSTRING_HSTRING_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define __FIIterator_1___FIKeyValuePair_2_HSTRING_HSTRING_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define __FIIterator_1___FIKeyValuePair_2_HSTRING_HSTRING_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 

#define __FIIterator_1___FIKeyValuePair_2_HSTRING_HSTRING_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 

#define __FIIterator_1___FIKeyValuePair_2_HSTRING_HSTRING_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 


#define __FIIterator_1___FIKeyValuePair_2_HSTRING_HSTRING_get_Current(This,current)	\
    ( (This)->lpVtbl -> get_Current(This,current) ) 

#define __FIIterator_1___FIKeyValuePair_2_HSTRING_HSTRING_get_HasCurrent(This,hasCurrent)	\
    ( (This)->lpVtbl -> get_HasCurrent(This,hasCurrent) ) 

#define __FIIterator_1___FIKeyValuePair_2_HSTRING_HSTRING_MoveNext(This,hasCurrent)	\
    ( (This)->lpVtbl -> MoveNext(This,hasCurrent) ) 

#define __FIIterator_1___FIKeyValuePair_2_HSTRING_HSTRING_GetMany(This,capacity,items,actual)	\
    ( (This)->lpVtbl -> GetMany(This,capacity,items,actual) ) 

#endif /* COBJMACROS */


#endif // ____FIIterator_1___FIKeyValuePair_2_HSTRING_HSTRING_INTERFACE_DEFINED__



#if !defined(____FIIterable_1___FIKeyValuePair_2_HSTRING_HSTRING_INTERFACE_DEFINED__)
#define ____FIIterable_1___FIKeyValuePair_2_HSTRING_HSTRING_INTERFACE_DEFINED__

typedef interface __FIIterable_1___FIKeyValuePair_2_HSTRING_HSTRING __FIIterable_1___FIKeyValuePair_2_HSTRING_HSTRING;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIIterable_1___FIKeyValuePair_2_HSTRING_HSTRING;

typedef  struct __FIIterable_1___FIKeyValuePair_2_HSTRING_HSTRINGVtbl
{
    BEGIN_INTERFACE

    HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
        __RPC__in __FIIterable_1___FIKeyValuePair_2_HSTRING_HSTRING * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);

    ULONG ( STDMETHODCALLTYPE *AddRef )(__RPC__in __FIIterable_1___FIKeyValuePair_2_HSTRING_HSTRING * This);

    ULONG ( STDMETHODCALLTYPE *Release )(__RPC__in __FIIterable_1___FIKeyValuePair_2_HSTRING_HSTRING * This);

    HRESULT ( STDMETHODCALLTYPE *GetIids )(__RPC__in __FIIterable_1___FIKeyValuePair_2_HSTRING_HSTRING * This,
                                           /* [out] */ __RPC__out ULONG *iidCount,
                                           /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);

    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(__RPC__in __FIIterable_1___FIKeyValuePair_2_HSTRING_HSTRING * This, /* [out] */ __RPC__deref_out_opt HSTRING *className);

    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(__RPC__in __FIIterable_1___FIKeyValuePair_2_HSTRING_HSTRING * This, /* [out] */ __RPC__out TrustLevel *trustLevel);

    HRESULT ( STDMETHODCALLTYPE *First )(__RPC__in __FIIterable_1___FIKeyValuePair_2_HSTRING_HSTRING * This, /* [retval][out] */ __RPC__deref_out_opt __FIIterator_1___FIKeyValuePair_2_HSTRING_HSTRING **first);

    END_INTERFACE
} __FIIterable_1___FIKeyValuePair_2_HSTRING_HSTRINGVtbl;

interface __FIIterable_1___FIKeyValuePair_2_HSTRING_HSTRING
{
    CONST_VTBL struct __FIIterable_1___FIKeyValuePair_2_HSTRING_HSTRINGVtbl *lpVtbl;
};

#ifdef COBJMACROS

#define __FIIterable_1___FIKeyValuePair_2_HSTRING_HSTRING_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define __FIIterable_1___FIKeyValuePair_2_HSTRING_HSTRING_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define __FIIterable_1___FIKeyValuePair_2_HSTRING_HSTRING_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define __FIIterable_1___FIKeyValuePair_2_HSTRING_HSTRING_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 

#define __FIIterable_1___FIKeyValuePair_2_HSTRING_HSTRING_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 

#define __FIIterable_1___FIKeyValuePair_2_HSTRING_HSTRING_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 


#define __FIIterable_1___FIKeyValuePair_2_HSTRING_HSTRING_First(This,first)	\
    ( (This)->lpVtbl -> First(This,first) ) 

#endif /* COBJMACROS */


#endif // ____FIIterable_1___FIKeyValuePair_2_HSTRING_HSTRING_INTERFACE_DEFINED__


#if !defined(____FIMapView_2_HSTRING_HSTRING_INTERFACE_DEFINED__)
#define ____FIMapView_2_HSTRING_HSTRING_INTERFACE_DEFINED__

typedef interface __FIMapView_2_HSTRING_HSTRING __FIMapView_2_HSTRING_HSTRING;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIMapView_2_HSTRING_HSTRING;

typedef struct __FIMapView_2_HSTRING_HSTRINGVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface )(__RPC__in __FIMapView_2_HSTRING_HSTRING * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);

    ULONG ( STDMETHODCALLTYPE *AddRef )(__RPC__in __FIMapView_2_HSTRING_HSTRING * This);

    ULONG ( STDMETHODCALLTYPE *Release )(__RPC__in __FIMapView_2_HSTRING_HSTRING * This);

    HRESULT ( STDMETHODCALLTYPE *GetIids )(__RPC__in __FIMapView_2_HSTRING_HSTRING * This,
        /* [out] */ __RPC__out ULONG *iidCount,
        /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);

    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(__RPC__in __FIMapView_2_HSTRING_HSTRING * This, /* [out] */ __RPC__deref_out_opt HSTRING *className);
    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(__RPC__in __FIMapView_2_HSTRING_HSTRING * This,/* [out] */ __RPC__out TrustLevel *trustLevel);

    HRESULT ( STDMETHODCALLTYPE *Lookup )(__RPC__in __FIMapView_2_HSTRING_HSTRING * This,
        /* [in] */ __RPC__in HSTRING key,
        /* [retval][out] */ __RPC__deref_out_opt HSTRING *value);
    /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Size )(__RPC__in __FIMapView_2_HSTRING_HSTRING * This, /* [retval][out] */ __RPC__out unsigned int *size);
    HRESULT ( STDMETHODCALLTYPE *HasKey )(__RPC__in __FIMapView_2_HSTRING_HSTRING * This, /* [in] */ __RPC__in HSTRING key, /* [retval][out] */ __RPC__out boolean *found);
    HRESULT ( STDMETHODCALLTYPE *Split )(__RPC__in __FIMapView_2_HSTRING_HSTRING * This,/* [out] */ __RPC__deref_out_opt __FIMapView_2_HSTRING_HSTRING **firstPartition,
        /* [out] */ __RPC__deref_out_opt __FIMapView_2_HSTRING_HSTRING **secondPartition);
    END_INTERFACE
} __FIMapView_2_HSTRING_HSTRINGVtbl;

interface __FIMapView_2_HSTRING_HSTRING
{
    CONST_VTBL struct __FIMapView_2_HSTRING_HSTRINGVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __FIMapView_2_HSTRING_HSTRING_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 
#define __FIMapView_2_HSTRING_HSTRING_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 
#define __FIMapView_2_HSTRING_HSTRING_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 

#define __FIMapView_2_HSTRING_HSTRING_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 
#define __FIMapView_2_HSTRING_HSTRING_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 
#define __FIMapView_2_HSTRING_HSTRING_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 

#define __FIMapView_2_HSTRING_HSTRING_Lookup(This,key,value)	\
    ( (This)->lpVtbl -> Lookup(This,key,value) ) 
#define __FIMapView_2_HSTRING_HSTRING_get_Size(This,size)	\
    ( (This)->lpVtbl -> get_Size(This,size) ) 
#define __FIMapView_2_HSTRING_HSTRING_HasKey(This,key,found)	\
    ( (This)->lpVtbl -> HasKey(This,key,found) ) 
#define __FIMapView_2_HSTRING_HSTRING_Split(This,firstPartition,secondPartition)	\
    ( (This)->lpVtbl -> Split(This,firstPartition,secondPartition) ) 
#endif /* COBJMACROS */


#endif // ____FIMapView_2_HSTRING_HSTRING_INTERFACE_DEFINED__



#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____FIIterator_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_INTERFACE_DEFINED__)
#define ____FIIterator_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_INTERFACE_DEFINED__

typedef interface __FIIterator_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor __FIIterator_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIIterator_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor;

typedef struct __FIIterator_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptorVtbl
{
    BEGIN_INTERFACE

    HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
        __RPC__in __FIIterator_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);
    ULONG ( STDMETHODCALLTYPE *AddRef )(__RPC__in __FIIterator_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor * This);
    ULONG ( STDMETHODCALLTYPE *Release )(__RPC__in __FIIterator_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor * This);
    HRESULT ( STDMETHODCALLTYPE *GetIids )(__RPC__in __FIIterator_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor * This,
        /* [out] */ __RPC__out ULONG *iidCount,
        /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);

    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(__RPC__in __FIIterator_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor * This, /* [out] */ __RPC__deref_out_opt HSTRING *className);
    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(__RPC__in __FIIterator_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor * This, /* [out] */ __RPC__out TrustLevel *trustLevel);

    /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Current )(__RPC__in __FIIterator_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor * This, /* [retval][out] */ __RPC__out __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor * *current);
    /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_HasCurrent )(__RPC__in __FIIterator_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor * This, /* [retval][out] */ __RPC__out boolean *hasCurrent);
    HRESULT ( STDMETHODCALLTYPE *MoveNext )(__RPC__in __FIIterator_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor * This, /* [retval][out] */ __RPC__out boolean *hasCurrent);
    HRESULT ( STDMETHODCALLTYPE *GetMany )(__RPC__in __FIIterator_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor * This,
        /* [in] */ unsigned int capacity,
        /* [size_is][length_is][out] */ __RPC__out_ecount_part(capacity, *actual) __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor * *items,
        /* [retval][out] */ __RPC__out unsigned int *actual);

    END_INTERFACE
} __FIIterator_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptorVtbl;

interface __FIIterator_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor
{
    CONST_VTBL struct __FIIterator_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptorVtbl *lpVtbl;
};



#ifdef COBJMACROS


#define __FIIterator_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define __FIIterator_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define __FIIterator_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define __FIIterator_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 

#define __FIIterator_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 

#define __FIIterator_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 


#define __FIIterator_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_get_Current(This,current)	\
    ( (This)->lpVtbl -> get_Current(This,current) ) 

#define __FIIterator_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_get_HasCurrent(This,hasCurrent)	\
    ( (This)->lpVtbl -> get_HasCurrent(This,hasCurrent) ) 

#define __FIIterator_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_MoveNext(This,hasCurrent)	\
    ( (This)->lpVtbl -> MoveNext(This,hasCurrent) ) 

#define __FIIterator_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_GetMany(This,capacity,items,actual)	\
    ( (This)->lpVtbl -> GetMany(This,capacity,items,actual) ) 

#endif /* COBJMACROS */


#endif // ____FIIterator_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_INTERFACE_DEFINED__

#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____FIIterable_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_INTERFACE_DEFINED__)
#define ____FIIterable_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_INTERFACE_DEFINED__

typedef interface __FIIterable_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor __FIIterable_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIIterable_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor;

typedef  struct __FIIterable_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptorVtbl
{
    BEGIN_INTERFACE

    HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
        __RPC__in __FIIterable_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);

    ULONG ( STDMETHODCALLTYPE *AddRef )(__RPC__in __FIIterable_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor * This);

    ULONG ( STDMETHODCALLTYPE *Release )(__RPC__in __FIIterable_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor * This);

    HRESULT ( STDMETHODCALLTYPE *GetIids )(__RPC__in __FIIterable_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor * This,
                                           /* [out] */ __RPC__out ULONG *iidCount,
                                           /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);

    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(__RPC__in __FIIterable_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor * This, /* [out] */ __RPC__deref_out_opt HSTRING *className);

    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(__RPC__in __FIIterable_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor * This, /* [out] */ __RPC__out TrustLevel *trustLevel);

    HRESULT ( STDMETHODCALLTYPE *First )(__RPC__in __FIIterable_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor * This, /* [retval][out] */ __RPC__deref_out_opt __FIIterator_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor **first);

    END_INTERFACE
} __FIIterable_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptorVtbl;

interface __FIIterable_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor
{
    CONST_VTBL struct __FIIterable_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptorVtbl *lpVtbl;
};

#ifdef COBJMACROS

#define __FIIterable_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define __FIIterable_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define __FIIterable_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define __FIIterable_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 

#define __FIIterable_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 

#define __FIIterable_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 


#define __FIIterable_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_First(This,first)	\
    ( (This)->lpVtbl -> First(This,first) ) 

#endif /* COBJMACROS */


#endif // ____FIIterable_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_INTERFACE_DEFINED__

#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____FIVectorView_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_INTERFACE_DEFINED__)
#define ____FIVectorView_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_INTERFACE_DEFINED__

typedef interface __FIVectorView_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor __FIVectorView_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIVectorView_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor;

typedef struct __FIVectorView_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptorVtbl
{
    BEGIN_INTERFACE

    HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
        __RPC__in __FIVectorView_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);

    ULONG ( STDMETHODCALLTYPE *AddRef )( __RPC__in __FIVectorView_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor * This);

    ULONG ( STDMETHODCALLTYPE *Release )( __RPC__in __FIVectorView_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor * This);

    HRESULT ( STDMETHODCALLTYPE *GetIids )( __RPC__in __FIVectorView_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor * This,
                                            /* [out] */ __RPC__out ULONG *iidCount,
                                            /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);

    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )( 
        __RPC__in __FIVectorView_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor * This,
            /* [out] */ __RPC__deref_out_opt HSTRING *className);

    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )( 
        __RPC__in __FIVectorView_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor * This,
            /* [out] */ __RPC__out TrustLevel *trustLevel);

    HRESULT ( STDMETHODCALLTYPE *GetAt )( 
                                         __RPC__in __FIVectorView_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor * This,
                                         /* [in] */ unsigned int index,
                                         /* [retval][out] */ __RPC__out __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor * *item);

        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Size )( 
            __RPC__in __FIVectorView_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor * This,
            /* [retval][out] */ __RPC__out unsigned int *size);

        HRESULT ( STDMETHODCALLTYPE *IndexOf )( 
                                               __RPC__in __FIVectorView_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor * This,
            /* [in] */ __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor * item,
            /* [out] */ __RPC__out unsigned int *index,
            /* [retval][out] */ __RPC__out boolean *found);

        HRESULT ( STDMETHODCALLTYPE *GetMany )( 
                                               __RPC__in __FIVectorView_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor * This,
            /* [in] */ unsigned int startIndex,
            /* [in] */ unsigned int capacity,
            /* [size_is][length_is][out] */ __RPC__out_ecount_part(capacity, *actual) __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor * *items,
            /* [retval][out] */ __RPC__out unsigned int *actual);

        END_INTERFACE
} __FIVectorView_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptorVtbl;

interface __FIVectorView_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor
{
    CONST_VTBL struct __FIVectorView_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptorVtbl *lpVtbl;
};



#ifdef COBJMACROS


#define __FIVectorView_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define __FIVectorView_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define __FIVectorView_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define __FIVectorView_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 

#define __FIVectorView_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 

#define __FIVectorView_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 


#define __FIVectorView_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_GetAt(This,index,item)	\
    ( (This)->lpVtbl -> GetAt(This,index,item) ) 

#define __FIVectorView_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_get_Size(This,size)	\
    ( (This)->lpVtbl -> get_Size(This,size) ) 

#define __FIVectorView_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_IndexOf(This,item,index,found)	\
    ( (This)->lpVtbl -> IndexOf(This,item,index,found) ) 

#define __FIVectorView_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_GetMany(This,startIndex,capacity,items,actual)	\
    ( (This)->lpVtbl -> GetMany(This,startIndex,capacity,items,actual) ) 

#endif /* COBJMACROS */



#endif // ____FIVectorView_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor_INTERFACE_DEFINED__

#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModel_INTERFACE_DEFINED__)
#define ____FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModel_INTERFACE_DEFINED__

typedef interface __FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModel __FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModel;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModel;

// Forward declare the async operation.
typedef interface __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModel __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModel;

typedef struct __FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModelVtbl
{
    BEGIN_INTERFACE

    HRESULT ( STDMETHODCALLTYPE *QueryInterface )(__RPC__in __FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModel * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);
    ULONG ( STDMETHODCALLTYPE *AddRef )(__RPC__in __FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModel * This);
    ULONG ( STDMETHODCALLTYPE *Release )(__RPC__in __FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModel * This);

    HRESULT ( STDMETHODCALLTYPE *Invoke )(__RPC__in __FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModel * This,/* [in] */ __RPC__in_opt __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModel *asyncInfo, /* [in] */ AsyncStatus status);
    END_INTERFACE
} __FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModelVtbl;

interface __FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModel
{
    CONST_VTBL struct __FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModelVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModel_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 
#define __FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModel_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 
#define __FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModel_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 

#define __FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModel_Invoke(This,asyncInfo,status)	\
    ( (This)->lpVtbl -> Invoke(This,asyncInfo,status) ) 
#endif /* COBJMACROS */


#endif // ____FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModel_INTERFACE_DEFINED__

#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModel_INTERFACE_DEFINED__)
#define ____FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModel_INTERFACE_DEFINED__

typedef interface __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModel __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModel;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModel;

typedef struct __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface )(__RPC__in __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModel * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);
    ULONG ( STDMETHODCALLTYPE *AddRef )(__RPC__in __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModel * This);
    ULONG ( STDMETHODCALLTYPE *Release )(__RPC__in __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModel * This);

    HRESULT ( STDMETHODCALLTYPE *GetIids )(__RPC__in __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModel * This,
        /* [out] */ __RPC__out ULONG *iidCount,
        /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);
    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(__RPC__in __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModel * This, /* [out] */ __RPC__deref_out_opt HSTRING *className);
    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(__RPC__in __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModel * This, /* [out] */ __RPC__out TrustLevel *trustLevel);

    /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_Completed )(__RPC__in __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModel * This, /* [in] */ __RPC__in_opt __FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModel *handler);
    /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Completed )(__RPC__in __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModel * This, /* [retval][out] */ __RPC__deref_out_opt __FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModel **handler);
    HRESULT ( STDMETHODCALLTYPE *GetResults )(__RPC__in __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModel * This, /* [retval][out] */ __RPC__out __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel * *results);
    END_INTERFACE
} __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelVtbl;

interface __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModel
{
    CONST_VTBL struct __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModel_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 
#define __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModel_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 
#define __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModel_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 

#define __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModel_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 
#define __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModel_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 
#define __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModel_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 

#define __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModel_put_Completed(This,handler)	\
    ( (This)->lpVtbl -> put_Completed(This,handler) ) 
#define __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModel_get_Completed(This,handler)	\
    ( (This)->lpVtbl -> get_Completed(This,handler) ) 
#define __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModel_GetResults(This,results)	\
    ( (This)->lpVtbl -> GetResults(This,results) ) 
#endif /* COBJMACROS */


#endif // ____FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModel_INTERFACE_DEFINED__

#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


#if !defined(____FIKeyValuePair_2_HSTRING_IInspectable_INTERFACE_DEFINED__)
#define ____FIKeyValuePair_2_HSTRING_IInspectable_INTERFACE_DEFINED__

typedef interface __FIKeyValuePair_2_HSTRING_IInspectable __FIKeyValuePair_2_HSTRING_IInspectable;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIKeyValuePair_2_HSTRING_IInspectable;

typedef struct __FIKeyValuePair_2_HSTRING_IInspectableVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface )(__RPC__in __FIKeyValuePair_2_HSTRING_IInspectable * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);

    ULONG ( STDMETHODCALLTYPE *AddRef )(__RPC__in __FIKeyValuePair_2_HSTRING_IInspectable * This);
    ULONG ( STDMETHODCALLTYPE *Release )(__RPC__in __FIKeyValuePair_2_HSTRING_IInspectable * This);
    HRESULT ( STDMETHODCALLTYPE *GetIids )(__RPC__in __FIKeyValuePair_2_HSTRING_IInspectable * This,
            /* [out] */ __RPC__out ULONG *iidCount,
            /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);
    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(__RPC__in __FIKeyValuePair_2_HSTRING_IInspectable * This, /* [out] */ __RPC__deref_out_opt HSTRING *className);
    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(__RPC__in __FIKeyValuePair_2_HSTRING_IInspectable * This, /* [out] */ __RPC__out TrustLevel *trustLevel);

    /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Key )(__RPC__in __FIKeyValuePair_2_HSTRING_IInspectable * This, /* [retval][out] */ __RPC__out HSTRING *key);
    /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Value )(__RPC__in __FIKeyValuePair_2_HSTRING_IInspectable * This, /* [retval][out] */ __RPC__deref_out_opt IInspectable * *value);
    END_INTERFACE
} __FIKeyValuePair_2_HSTRING_IInspectableVtbl;

interface __FIKeyValuePair_2_HSTRING_IInspectable
{
    CONST_VTBL struct __FIKeyValuePair_2_HSTRING_IInspectableVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __FIKeyValuePair_2_HSTRING_IInspectable_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define __FIKeyValuePair_2_HSTRING_IInspectable_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define __FIKeyValuePair_2_HSTRING_IInspectable_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define __FIKeyValuePair_2_HSTRING_IInspectable_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 

#define __FIKeyValuePair_2_HSTRING_IInspectable_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 

#define __FIKeyValuePair_2_HSTRING_IInspectable_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 


#define __FIKeyValuePair_2_HSTRING_IInspectable_get_Key(This,key)	\
    ( (This)->lpVtbl -> get_Key(This,key) ) 

#define __FIKeyValuePair_2_HSTRING_IInspectable_get_Value(This,value)	\
    ( (This)->lpVtbl -> get_Value(This,value) ) 
#endif /* COBJMACROS */


#endif // ____FIKeyValuePair_2_HSTRING_IInspectable_INTERFACE_DEFINED__



#if !defined(____FIIterator_1___FIKeyValuePair_2_HSTRING_IInspectable_INTERFACE_DEFINED__)
#define ____FIIterator_1___FIKeyValuePair_2_HSTRING_IInspectable_INTERFACE_DEFINED__

typedef interface __FIIterator_1___FIKeyValuePair_2_HSTRING_IInspectable __FIIterator_1___FIKeyValuePair_2_HSTRING_IInspectable;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIIterator_1___FIKeyValuePair_2_HSTRING_IInspectable;

typedef struct __FIIterator_1___FIKeyValuePair_2_HSTRING_IInspectableVtbl
{
    BEGIN_INTERFACE

    HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
        __RPC__in __FIIterator_1___FIKeyValuePair_2_HSTRING_IInspectable * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);
    ULONG ( STDMETHODCALLTYPE *AddRef )(__RPC__in __FIIterator_1___FIKeyValuePair_2_HSTRING_IInspectable * This);
    ULONG ( STDMETHODCALLTYPE *Release )(__RPC__in __FIIterator_1___FIKeyValuePair_2_HSTRING_IInspectable * This);
    HRESULT ( STDMETHODCALLTYPE *GetIids )(__RPC__in __FIIterator_1___FIKeyValuePair_2_HSTRING_IInspectable * This,
        /* [out] */ __RPC__out ULONG *iidCount,
        /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);

    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(__RPC__in __FIIterator_1___FIKeyValuePair_2_HSTRING_IInspectable * This, /* [out] */ __RPC__deref_out_opt HSTRING *className);
    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(__RPC__in __FIIterator_1___FIKeyValuePair_2_HSTRING_IInspectable * This, /* [out] */ __RPC__out TrustLevel *trustLevel);

    /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Current )(__RPC__in __FIIterator_1___FIKeyValuePair_2_HSTRING_IInspectable * This, /* [retval][out] */ __RPC__out __FIKeyValuePair_2_HSTRING_IInspectable * *current);
    /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_HasCurrent )(__RPC__in __FIIterator_1___FIKeyValuePair_2_HSTRING_IInspectable * This, /* [retval][out] */ __RPC__out boolean *hasCurrent);
    HRESULT ( STDMETHODCALLTYPE *MoveNext )(__RPC__in __FIIterator_1___FIKeyValuePair_2_HSTRING_IInspectable * This, /* [retval][out] */ __RPC__out boolean *hasCurrent);
    HRESULT ( STDMETHODCALLTYPE *GetMany )(__RPC__in __FIIterator_1___FIKeyValuePair_2_HSTRING_IInspectable * This,
        /* [in] */ unsigned int capacity,
        /* [size_is][length_is][out] */ __RPC__out_ecount_part(capacity, *actual) __FIKeyValuePair_2_HSTRING_IInspectable * *items,
        /* [retval][out] */ __RPC__out unsigned int *actual);

    END_INTERFACE
} __FIIterator_1___FIKeyValuePair_2_HSTRING_IInspectableVtbl;

interface __FIIterator_1___FIKeyValuePair_2_HSTRING_IInspectable
{
    CONST_VTBL struct __FIIterator_1___FIKeyValuePair_2_HSTRING_IInspectableVtbl *lpVtbl;
};



#ifdef COBJMACROS


#define __FIIterator_1___FIKeyValuePair_2_HSTRING_IInspectable_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define __FIIterator_1___FIKeyValuePair_2_HSTRING_IInspectable_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define __FIIterator_1___FIKeyValuePair_2_HSTRING_IInspectable_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define __FIIterator_1___FIKeyValuePair_2_HSTRING_IInspectable_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 

#define __FIIterator_1___FIKeyValuePair_2_HSTRING_IInspectable_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 

#define __FIIterator_1___FIKeyValuePair_2_HSTRING_IInspectable_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 


#define __FIIterator_1___FIKeyValuePair_2_HSTRING_IInspectable_get_Current(This,current)	\
    ( (This)->lpVtbl -> get_Current(This,current) ) 

#define __FIIterator_1___FIKeyValuePair_2_HSTRING_IInspectable_get_HasCurrent(This,hasCurrent)	\
    ( (This)->lpVtbl -> get_HasCurrent(This,hasCurrent) ) 

#define __FIIterator_1___FIKeyValuePair_2_HSTRING_IInspectable_MoveNext(This,hasCurrent)	\
    ( (This)->lpVtbl -> MoveNext(This,hasCurrent) ) 

#define __FIIterator_1___FIKeyValuePair_2_HSTRING_IInspectable_GetMany(This,capacity,items,actual)	\
    ( (This)->lpVtbl -> GetMany(This,capacity,items,actual) ) 

#endif /* COBJMACROS */


#endif // ____FIIterator_1___FIKeyValuePair_2_HSTRING_IInspectable_INTERFACE_DEFINED__



#if !defined(____FIIterable_1___FIKeyValuePair_2_HSTRING_IInspectable_INTERFACE_DEFINED__)
#define ____FIIterable_1___FIKeyValuePair_2_HSTRING_IInspectable_INTERFACE_DEFINED__

typedef interface __FIIterable_1___FIKeyValuePair_2_HSTRING_IInspectable __FIIterable_1___FIKeyValuePair_2_HSTRING_IInspectable;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIIterable_1___FIKeyValuePair_2_HSTRING_IInspectable;

typedef  struct __FIIterable_1___FIKeyValuePair_2_HSTRING_IInspectableVtbl
{
    BEGIN_INTERFACE

    HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
        __RPC__in __FIIterable_1___FIKeyValuePair_2_HSTRING_IInspectable * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);

    ULONG ( STDMETHODCALLTYPE *AddRef )(__RPC__in __FIIterable_1___FIKeyValuePair_2_HSTRING_IInspectable * This);

    ULONG ( STDMETHODCALLTYPE *Release )(__RPC__in __FIIterable_1___FIKeyValuePair_2_HSTRING_IInspectable * This);

    HRESULT ( STDMETHODCALLTYPE *GetIids )(__RPC__in __FIIterable_1___FIKeyValuePair_2_HSTRING_IInspectable * This,
                                           /* [out] */ __RPC__out ULONG *iidCount,
                                           /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);

    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(__RPC__in __FIIterable_1___FIKeyValuePair_2_HSTRING_IInspectable * This, /* [out] */ __RPC__deref_out_opt HSTRING *className);

    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(__RPC__in __FIIterable_1___FIKeyValuePair_2_HSTRING_IInspectable * This, /* [out] */ __RPC__out TrustLevel *trustLevel);

    HRESULT ( STDMETHODCALLTYPE *First )(__RPC__in __FIIterable_1___FIKeyValuePair_2_HSTRING_IInspectable * This, /* [retval][out] */ __RPC__deref_out_opt __FIIterator_1___FIKeyValuePair_2_HSTRING_IInspectable **first);

    END_INTERFACE
} __FIIterable_1___FIKeyValuePair_2_HSTRING_IInspectableVtbl;

interface __FIIterable_1___FIKeyValuePair_2_HSTRING_IInspectable
{
    CONST_VTBL struct __FIIterable_1___FIKeyValuePair_2_HSTRING_IInspectableVtbl *lpVtbl;
};

#ifdef COBJMACROS

#define __FIIterable_1___FIKeyValuePair_2_HSTRING_IInspectable_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define __FIIterable_1___FIKeyValuePair_2_HSTRING_IInspectable_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define __FIIterable_1___FIKeyValuePair_2_HSTRING_IInspectable_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define __FIIterable_1___FIKeyValuePair_2_HSTRING_IInspectable_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 

#define __FIIterable_1___FIKeyValuePair_2_HSTRING_IInspectable_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 

#define __FIIterable_1___FIKeyValuePair_2_HSTRING_IInspectable_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 


#define __FIIterable_1___FIKeyValuePair_2_HSTRING_IInspectable_First(This,first)	\
    ( (This)->lpVtbl -> First(This,first) ) 

#endif /* COBJMACROS */


#endif // ____FIIterable_1___FIKeyValuePair_2_HSTRING_IInspectable_INTERFACE_DEFINED__



#if !defined(____FIMapView_2_HSTRING_IInspectable_INTERFACE_DEFINED__)
#define ____FIMapView_2_HSTRING_IInspectable_INTERFACE_DEFINED__

typedef interface __FIMapView_2_HSTRING_IInspectable __FIMapView_2_HSTRING_IInspectable;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIMapView_2_HSTRING_IInspectable;

typedef struct __FIMapView_2_HSTRING_IInspectableVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface )(__RPC__in __FIMapView_2_HSTRING_IInspectable * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);

    ULONG ( STDMETHODCALLTYPE *AddRef )(__RPC__in __FIMapView_2_HSTRING_IInspectable * This);

    ULONG ( STDMETHODCALLTYPE *Release )(__RPC__in __FIMapView_2_HSTRING_IInspectable * This);

    HRESULT ( STDMETHODCALLTYPE *GetIids )(__RPC__in __FIMapView_2_HSTRING_IInspectable * This,
        /* [out] */ __RPC__out ULONG *iidCount,
        /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);

    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(__RPC__in __FIMapView_2_HSTRING_IInspectable * This, /* [out] */ __RPC__deref_out_opt HSTRING *className);
    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(__RPC__in __FIMapView_2_HSTRING_IInspectable * This,/* [out] */ __RPC__out TrustLevel *trustLevel);

    HRESULT ( STDMETHODCALLTYPE *Lookup )(__RPC__in __FIMapView_2_HSTRING_IInspectable * This,
        /* [in] */ __RPC__in HSTRING key,
        /* [retval][out] */ __RPC__deref_out_opt IInspectable * *value);
    /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Size )(__RPC__in __FIMapView_2_HSTRING_IInspectable * This, /* [retval][out] */ __RPC__out unsigned int *size);
    HRESULT ( STDMETHODCALLTYPE *HasKey )(__RPC__in __FIMapView_2_HSTRING_IInspectable * This, /* [in] */ __RPC__in HSTRING key, /* [retval][out] */ __RPC__out boolean *found);
    HRESULT ( STDMETHODCALLTYPE *Split )(__RPC__in __FIMapView_2_HSTRING_IInspectable * This,/* [out] */ __RPC__deref_out_opt __FIMapView_2_HSTRING_IInspectable **firstPartition,
        /* [out] */ __RPC__deref_out_opt __FIMapView_2_HSTRING_IInspectable **secondPartition);
    END_INTERFACE
} __FIMapView_2_HSTRING_IInspectableVtbl;

interface __FIMapView_2_HSTRING_IInspectable
{
    CONST_VTBL struct __FIMapView_2_HSTRING_IInspectableVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __FIMapView_2_HSTRING_IInspectable_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 
#define __FIMapView_2_HSTRING_IInspectable_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 
#define __FIMapView_2_HSTRING_IInspectable_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 

#define __FIMapView_2_HSTRING_IInspectable_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 
#define __FIMapView_2_HSTRING_IInspectable_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 
#define __FIMapView_2_HSTRING_IInspectable_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 

#define __FIMapView_2_HSTRING_IInspectable_Lookup(This,key,value)	\
    ( (This)->lpVtbl -> Lookup(This,key,value) ) 
#define __FIMapView_2_HSTRING_IInspectable_get_Size(This,size)	\
    ( (This)->lpVtbl -> get_Size(This,size) ) 
#define __FIMapView_2_HSTRING_IInspectable_HasKey(This,key,found)	\
    ( (This)->lpVtbl -> HasKey(This,key,found) ) 
#define __FIMapView_2_HSTRING_IInspectable_Split(This,firstPartition,secondPartition)	\
    ( (This)->lpVtbl -> Split(This,firstPartition,secondPartition) ) 
#endif /* COBJMACROS */


#endif // ____FIMapView_2_HSTRING_IInspectable_INTERFACE_DEFINED__



#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult_INTERFACE_DEFINED__)
#define ____FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult_INTERFACE_DEFINED__

typedef interface __FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult __FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult;

// Forward declare the async operation.
typedef interface __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult;

typedef struct __FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResultVtbl
{
    BEGIN_INTERFACE

    HRESULT ( STDMETHODCALLTYPE *QueryInterface )(__RPC__in __FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);
    ULONG ( STDMETHODCALLTYPE *AddRef )(__RPC__in __FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult * This);
    ULONG ( STDMETHODCALLTYPE *Release )(__RPC__in __FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult * This);

    HRESULT ( STDMETHODCALLTYPE *Invoke )(__RPC__in __FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult * This,/* [in] */ __RPC__in_opt __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult *asyncInfo, /* [in] */ AsyncStatus status);
    END_INTERFACE
} __FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResultVtbl;

interface __FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult
{
    CONST_VTBL struct __FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResultVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 
#define __FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 
#define __FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 

#define __FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult_Invoke(This,asyncInfo,status)	\
    ( (This)->lpVtbl -> Invoke(This,asyncInfo,status) ) 
#endif /* COBJMACROS */


#endif // ____FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult_INTERFACE_DEFINED__

#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult_INTERFACE_DEFINED__)
#define ____FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult_INTERFACE_DEFINED__

typedef interface __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult;

typedef struct __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResultVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface )(__RPC__in __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);
    ULONG ( STDMETHODCALLTYPE *AddRef )(__RPC__in __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult * This);
    ULONG ( STDMETHODCALLTYPE *Release )(__RPC__in __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult * This);

    HRESULT ( STDMETHODCALLTYPE *GetIids )(__RPC__in __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult * This,
        /* [out] */ __RPC__out ULONG *iidCount,
        /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);
    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(__RPC__in __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult * This, /* [out] */ __RPC__deref_out_opt HSTRING *className);
    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(__RPC__in __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult * This, /* [out] */ __RPC__out TrustLevel *trustLevel);

    /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_Completed )(__RPC__in __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult * This, /* [in] */ __RPC__in_opt __FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult *handler);
    /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Completed )(__RPC__in __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult * This, /* [retval][out] */ __RPC__deref_out_opt __FIAsyncOperationCompletedHandler_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult **handler);
    HRESULT ( STDMETHODCALLTYPE *GetResults )(__RPC__in __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult * This, /* [retval][out] */ __RPC__out __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelEvaluationResult * *results);
    END_INTERFACE
} __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResultVtbl;

interface __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult
{
    CONST_VTBL struct __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResultVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 
#define __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 
#define __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 

#define __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 
#define __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 
#define __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 

#define __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult_put_Completed(This,handler)	\
    ( (This)->lpVtbl -> put_Completed(This,handler) ) 
#define __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult_get_Completed(This,handler)	\
    ( (This)->lpVtbl -> get_Completed(This,handler) ) 
#define __FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult_GetResults(This,results)	\
    ( (This)->lpVtbl -> GetResults(This,results) ) 
#endif /* COBJMACROS */


#endif // ____FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult_INTERFACE_DEFINED__

#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


#if !defined(____FIMap_2_HSTRING_IInspectable_INTERFACE_DEFINED__)
#define ____FIMap_2_HSTRING_IInspectable_INTERFACE_DEFINED__

typedef interface __FIMap_2_HSTRING_IInspectable __FIMap_2_HSTRING_IInspectable;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIMap_2_HSTRING_IInspectable;

typedef struct __FIMap_2_HSTRING_IInspectableVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface )(__RPC__in __FIMap_2_HSTRING_IInspectable * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);
    ULONG ( STDMETHODCALLTYPE *AddRef )(__RPC__in __FIMap_2_HSTRING_IInspectable * This);
    ULONG ( STDMETHODCALLTYPE *Release )(__RPC__in __FIMap_2_HSTRING_IInspectable * This);

    HRESULT ( STDMETHODCALLTYPE *GetIids )(__RPC__in __FIMap_2_HSTRING_IInspectable * This,
        /* [out] */ __RPC__out ULONG *iidCount,
        /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);
    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(__RPC__in __FIMap_2_HSTRING_IInspectable * This, /* [out] */ __RPC__deref_out_opt HSTRING *className);
    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(__RPC__in __FIMap_2_HSTRING_IInspectable * This, /* [out] */ __RPC__out TrustLevel *trustLevel);

    HRESULT ( STDMETHODCALLTYPE *Lookup )(__RPC__in __FIMap_2_HSTRING_IInspectable * This,
        /* [in] */ HSTRING key,
        /* [retval][out] */ __RPC__deref_out_opt IInspectable * **value);
    /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Size )(__RPC__in __FIMap_2_HSTRING_IInspectable * This, /* [retval][out] */ __RPC__out unsigned int *size);
    HRESULT ( STDMETHODCALLTYPE *HasKey )(__RPC__in __FIMap_2_HSTRING_IInspectable * This, /* [in] */ HSTRING key, /* [retval][out] */ __RPC__out boolean *found);
    HRESULT ( STDMETHODCALLTYPE *GetView )(__RPC__in __FIMap_2_HSTRING_IInspectable * This, /* [retval][out] */ __RPC__deref_out_opt __FIMapView_2_HSTRING_IInspectable **view);
    HRESULT ( STDMETHODCALLTYPE *Insert )(__RPC__in __FIMap_2_HSTRING_IInspectable * This,
        /* [in] */ HSTRING key,
        /* [in] */ __RPC__in_opt IInspectable * *value,
        /* [retval][out] */ __RPC__out boolean *replaced);
    HRESULT ( STDMETHODCALLTYPE *Remove )(__RPC__in __FIMap_2_HSTRING_IInspectable * This,/* [in] */ HSTRING key);
    HRESULT ( STDMETHODCALLTYPE *Clear )(__RPC__in __FIMap_2_HSTRING_IInspectable * This);
    END_INTERFACE
} __FIMap_2_HSTRING_IInspectableVtbl;

interface __FIMap_2_HSTRING_IInspectable
{
    CONST_VTBL struct __FIMap_2_HSTRING_IInspectableVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __FIMap_2_HSTRING_IInspectable_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define __FIMap_2_HSTRING_IInspectable_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define __FIMap_2_HSTRING_IInspectable_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define __FIMap_2_HSTRING_IInspectable_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 

#define __FIMap_2_HSTRING_IInspectable_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 

#define __FIMap_2_HSTRING_IInspectable_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 


#define __FIMap_2_HSTRING_IInspectable_Lookup(This,key,value)	\
    ( (This)->lpVtbl -> Lookup(This,key,value) ) 

#define __FIMap_2_HSTRING_IInspectable_get_Size(This,size)	\
    ( (This)->lpVtbl -> get_Size(This,size) ) 

#define __FIMap_2_HSTRING_IInspectable_HasKey(This,key,found)	\
    ( (This)->lpVtbl -> HasKey(This,key,found) ) 

#define __FIMap_2_HSTRING_IInspectable_GetView(This,view)	\
    ( (This)->lpVtbl -> GetView(This,view) ) 

#define __FIMap_2_HSTRING_IInspectable_Insert(This,key,value,replaced)	\
    ( (This)->lpVtbl -> Insert(This,key,value,replaced) ) 

#define __FIMap_2_HSTRING_IInspectable_Remove(This,key)	\
    ( (This)->lpVtbl -> Remove(This,key) ) 

#define __FIMap_2_HSTRING_IInspectable_Clear(This)	\
    ( (This)->lpVtbl -> Clear(This) ) 
#endif /* COBJMACROS */



#endif // ____FIMap_2_HSTRING_IInspectable_INTERFACE_DEFINED__


#if !defined(____FIIterator_1___z__zint64_INTERFACE_DEFINED__)
#define ____FIIterator_1___z__zint64_INTERFACE_DEFINED__

typedef interface __FIIterator_1___z__zint64 __FIIterator_1___z__zint64;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIIterator_1___z__zint64;

typedef struct __FIIterator_1___z__zint64Vtbl
{
    BEGIN_INTERFACE

    HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
        __RPC__in __FIIterator_1___z__zint64 * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);
    ULONG ( STDMETHODCALLTYPE *AddRef )(__RPC__in __FIIterator_1___z__zint64 * This);
    ULONG ( STDMETHODCALLTYPE *Release )(__RPC__in __FIIterator_1___z__zint64 * This);
    HRESULT ( STDMETHODCALLTYPE *GetIids )(__RPC__in __FIIterator_1___z__zint64 * This,
        /* [out] */ __RPC__out ULONG *iidCount,
        /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);

    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(__RPC__in __FIIterator_1___z__zint64 * This, /* [out] */ __RPC__deref_out_opt HSTRING *className);
    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(__RPC__in __FIIterator_1___z__zint64 * This, /* [out] */ __RPC__out TrustLevel *trustLevel);

    /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Current )(__RPC__in __FIIterator_1___z__zint64 * This, /* [retval][out] */ __RPC__out __int64 *current);
    /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_HasCurrent )(__RPC__in __FIIterator_1___z__zint64 * This, /* [retval][out] */ __RPC__out boolean *hasCurrent);
    HRESULT ( STDMETHODCALLTYPE *MoveNext )(__RPC__in __FIIterator_1___z__zint64 * This, /* [retval][out] */ __RPC__out boolean *hasCurrent);
    HRESULT ( STDMETHODCALLTYPE *GetMany )(__RPC__in __FIIterator_1___z__zint64 * This,
        /* [in] */ unsigned int capacity,
        /* [size_is][length_is][out] */ __RPC__out_ecount_part(capacity, *actual) __int64 *items,
        /* [retval][out] */ __RPC__out unsigned int *actual);

    END_INTERFACE
} __FIIterator_1___z__zint64Vtbl;

interface __FIIterator_1___z__zint64
{
    CONST_VTBL struct __FIIterator_1___z__zint64Vtbl *lpVtbl;
};



#ifdef COBJMACROS


#define __FIIterator_1___z__zint64_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define __FIIterator_1___z__zint64_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define __FIIterator_1___z__zint64_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define __FIIterator_1___z__zint64_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 

#define __FIIterator_1___z__zint64_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 

#define __FIIterator_1___z__zint64_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 


#define __FIIterator_1___z__zint64_get_Current(This,current)	\
    ( (This)->lpVtbl -> get_Current(This,current) ) 

#define __FIIterator_1___z__zint64_get_HasCurrent(This,hasCurrent)	\
    ( (This)->lpVtbl -> get_HasCurrent(This,hasCurrent) ) 

#define __FIIterator_1___z__zint64_MoveNext(This,hasCurrent)	\
    ( (This)->lpVtbl -> MoveNext(This,hasCurrent) ) 

#define __FIIterator_1___z__zint64_GetMany(This,capacity,items,actual)	\
    ( (This)->lpVtbl -> GetMany(This,capacity,items,actual) ) 

#endif /* COBJMACROS */


#endif // ____FIIterator_1___z__zint64_INTERFACE_DEFINED__


#if !defined(____FIIterable_1___z__zint64_INTERFACE_DEFINED__)
#define ____FIIterable_1___z__zint64_INTERFACE_DEFINED__

typedef interface __FIIterable_1___z__zint64 __FIIterable_1___z__zint64;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIIterable_1___z__zint64;

typedef  struct __FIIterable_1___z__zint64Vtbl
{
    BEGIN_INTERFACE

    HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
        __RPC__in __FIIterable_1___z__zint64 * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);

    ULONG ( STDMETHODCALLTYPE *AddRef )(__RPC__in __FIIterable_1___z__zint64 * This);

    ULONG ( STDMETHODCALLTYPE *Release )(__RPC__in __FIIterable_1___z__zint64 * This);

    HRESULT ( STDMETHODCALLTYPE *GetIids )(__RPC__in __FIIterable_1___z__zint64 * This,
                                           /* [out] */ __RPC__out ULONG *iidCount,
                                           /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);

    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(__RPC__in __FIIterable_1___z__zint64 * This, /* [out] */ __RPC__deref_out_opt HSTRING *className);

    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(__RPC__in __FIIterable_1___z__zint64 * This, /* [out] */ __RPC__out TrustLevel *trustLevel);

    HRESULT ( STDMETHODCALLTYPE *First )(__RPC__in __FIIterable_1___z__zint64 * This, /* [retval][out] */ __RPC__deref_out_opt __FIIterator_1___z__zint64 **first);

    END_INTERFACE
} __FIIterable_1___z__zint64Vtbl;

interface __FIIterable_1___z__zint64
{
    CONST_VTBL struct __FIIterable_1___z__zint64Vtbl *lpVtbl;
};

#ifdef COBJMACROS

#define __FIIterable_1___z__zint64_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define __FIIterable_1___z__zint64_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define __FIIterable_1___z__zint64_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define __FIIterable_1___z__zint64_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 

#define __FIIterable_1___z__zint64_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 

#define __FIIterable_1___z__zint64_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 


#define __FIIterable_1___z__zint64_First(This,first)	\
    ( (This)->lpVtbl -> First(This,first) ) 

#endif /* COBJMACROS */


#endif // ____FIIterable_1___z__zint64_INTERFACE_DEFINED__


#if !defined(____FIVectorView_1___z__zint64_INTERFACE_DEFINED__)
#define ____FIVectorView_1___z__zint64_INTERFACE_DEFINED__

typedef interface __FIVectorView_1___z__zint64 __FIVectorView_1___z__zint64;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIVectorView_1___z__zint64;

typedef struct __FIVectorView_1___z__zint64Vtbl
{
    BEGIN_INTERFACE

    HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
        __RPC__in __FIVectorView_1___z__zint64 * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);

    ULONG ( STDMETHODCALLTYPE *AddRef )( __RPC__in __FIVectorView_1___z__zint64 * This);

    ULONG ( STDMETHODCALLTYPE *Release )( __RPC__in __FIVectorView_1___z__zint64 * This);

    HRESULT ( STDMETHODCALLTYPE *GetIids )( __RPC__in __FIVectorView_1___z__zint64 * This,
                                            /* [out] */ __RPC__out ULONG *iidCount,
                                            /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);

    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )( 
        __RPC__in __FIVectorView_1___z__zint64 * This,
            /* [out] */ __RPC__deref_out_opt HSTRING *className);

    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )( 
        __RPC__in __FIVectorView_1___z__zint64 * This,
            /* [out] */ __RPC__out TrustLevel *trustLevel);

    HRESULT ( STDMETHODCALLTYPE *GetAt )( 
                                         __RPC__in __FIVectorView_1___z__zint64 * This,
                                         /* [in] */ unsigned int index,
                                         /* [retval][out] */ __RPC__out __int64 *item);

        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Size )( 
            __RPC__in __FIVectorView_1___z__zint64 * This,
            /* [retval][out] */ __RPC__out unsigned int *size);

        HRESULT ( STDMETHODCALLTYPE *IndexOf )( 
                                               __RPC__in __FIVectorView_1___z__zint64 * This,
            /* [in] */ __int64 item,
            /* [out] */ __RPC__out unsigned int *index,
            /* [retval][out] */ __RPC__out boolean *found);

        HRESULT ( STDMETHODCALLTYPE *GetMany )( 
                                               __RPC__in __FIVectorView_1___z__zint64 * This,
            /* [in] */ unsigned int startIndex,
            /* [in] */ unsigned int capacity,
            /* [size_is][length_is][out] */ __RPC__out_ecount_part(capacity, *actual) __int64 *items,
            /* [retval][out] */ __RPC__out unsigned int *actual);

        END_INTERFACE
} __FIVectorView_1___z__zint64Vtbl;

interface __FIVectorView_1___z__zint64
{
    CONST_VTBL struct __FIVectorView_1___z__zint64Vtbl *lpVtbl;
};



#ifdef COBJMACROS


#define __FIVectorView_1___z__zint64_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define __FIVectorView_1___z__zint64_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define __FIVectorView_1___z__zint64_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define __FIVectorView_1___z__zint64_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 

#define __FIVectorView_1___z__zint64_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 

#define __FIVectorView_1___z__zint64_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 


#define __FIVectorView_1___z__zint64_GetAt(This,index,item)	\
    ( (This)->lpVtbl -> GetAt(This,index,item) ) 

#define __FIVectorView_1___z__zint64_get_Size(This,size)	\
    ( (This)->lpVtbl -> get_Size(This,size) ) 

#define __FIVectorView_1___z__zint64_IndexOf(This,item,index,found)	\
    ( (This)->lpVtbl -> IndexOf(This,item,index,found) ) 

#define __FIVectorView_1___z__zint64_GetMany(This,startIndex,capacity,items,actual)	\
    ( (This)->lpVtbl -> GetMany(This,startIndex,capacity,items,actual) ) 

#endif /* COBJMACROS */



#endif // ____FIVectorView_1___z__zint64_INTERFACE_DEFINED__


#if !defined(____FIIterator_1_float_INTERFACE_DEFINED__)
#define ____FIIterator_1_float_INTERFACE_DEFINED__

typedef interface __FIIterator_1_float __FIIterator_1_float;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIIterator_1_float;

typedef struct __FIIterator_1_floatVtbl
{
    BEGIN_INTERFACE

    HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
        __RPC__in __FIIterator_1_float * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);
    ULONG ( STDMETHODCALLTYPE *AddRef )(__RPC__in __FIIterator_1_float * This);
    ULONG ( STDMETHODCALLTYPE *Release )(__RPC__in __FIIterator_1_float * This);
    HRESULT ( STDMETHODCALLTYPE *GetIids )(__RPC__in __FIIterator_1_float * This,
        /* [out] */ __RPC__out ULONG *iidCount,
        /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);

    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(__RPC__in __FIIterator_1_float * This, /* [out] */ __RPC__deref_out_opt HSTRING *className);
    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(__RPC__in __FIIterator_1_float * This, /* [out] */ __RPC__out TrustLevel *trustLevel);

    /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Current )(__RPC__in __FIIterator_1_float * This, /* [retval][out] */ __RPC__out float *current);
    /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_HasCurrent )(__RPC__in __FIIterator_1_float * This, /* [retval][out] */ __RPC__out boolean *hasCurrent);
    HRESULT ( STDMETHODCALLTYPE *MoveNext )(__RPC__in __FIIterator_1_float * This, /* [retval][out] */ __RPC__out boolean *hasCurrent);
    HRESULT ( STDMETHODCALLTYPE *GetMany )(__RPC__in __FIIterator_1_float * This,
        /* [in] */ unsigned int capacity,
        /* [size_is][length_is][out] */ __RPC__out_ecount_part(capacity, *actual) float *items,
        /* [retval][out] */ __RPC__out unsigned int *actual);

    END_INTERFACE
} __FIIterator_1_floatVtbl;

interface __FIIterator_1_float
{
    CONST_VTBL struct __FIIterator_1_floatVtbl *lpVtbl;
};



#ifdef COBJMACROS


#define __FIIterator_1_float_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define __FIIterator_1_float_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define __FIIterator_1_float_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define __FIIterator_1_float_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 

#define __FIIterator_1_float_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 

#define __FIIterator_1_float_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 


#define __FIIterator_1_float_get_Current(This,current)	\
    ( (This)->lpVtbl -> get_Current(This,current) ) 

#define __FIIterator_1_float_get_HasCurrent(This,hasCurrent)	\
    ( (This)->lpVtbl -> get_HasCurrent(This,hasCurrent) ) 

#define __FIIterator_1_float_MoveNext(This,hasCurrent)	\
    ( (This)->lpVtbl -> MoveNext(This,hasCurrent) ) 

#define __FIIterator_1_float_GetMany(This,capacity,items,actual)	\
    ( (This)->lpVtbl -> GetMany(This,capacity,items,actual) ) 

#endif /* COBJMACROS */


#endif // ____FIIterator_1_float_INTERFACE_DEFINED__


#if !defined(____FIIterable_1_float_INTERFACE_DEFINED__)
#define ____FIIterable_1_float_INTERFACE_DEFINED__

typedef interface __FIIterable_1_float __FIIterable_1_float;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIIterable_1_float;

typedef  struct __FIIterable_1_floatVtbl
{
    BEGIN_INTERFACE

    HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
        __RPC__in __FIIterable_1_float * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);

    ULONG ( STDMETHODCALLTYPE *AddRef )(__RPC__in __FIIterable_1_float * This);

    ULONG ( STDMETHODCALLTYPE *Release )(__RPC__in __FIIterable_1_float * This);

    HRESULT ( STDMETHODCALLTYPE *GetIids )(__RPC__in __FIIterable_1_float * This,
                                           /* [out] */ __RPC__out ULONG *iidCount,
                                           /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);

    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(__RPC__in __FIIterable_1_float * This, /* [out] */ __RPC__deref_out_opt HSTRING *className);

    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(__RPC__in __FIIterable_1_float * This, /* [out] */ __RPC__out TrustLevel *trustLevel);

    HRESULT ( STDMETHODCALLTYPE *First )(__RPC__in __FIIterable_1_float * This, /* [retval][out] */ __RPC__deref_out_opt __FIIterator_1_float **first);

    END_INTERFACE
} __FIIterable_1_floatVtbl;

interface __FIIterable_1_float
{
    CONST_VTBL struct __FIIterable_1_floatVtbl *lpVtbl;
};

#ifdef COBJMACROS

#define __FIIterable_1_float_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define __FIIterable_1_float_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define __FIIterable_1_float_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define __FIIterable_1_float_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 

#define __FIIterable_1_float_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 

#define __FIIterable_1_float_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 


#define __FIIterable_1_float_First(This,first)	\
    ( (This)->lpVtbl -> First(This,first) ) 

#endif /* COBJMACROS */


#endif // ____FIIterable_1_float_INTERFACE_DEFINED__


#if !defined(____FIVectorView_1_float_INTERFACE_DEFINED__)
#define ____FIVectorView_1_float_INTERFACE_DEFINED__

typedef interface __FIVectorView_1_float __FIVectorView_1_float;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIVectorView_1_float;

typedef struct __FIVectorView_1_floatVtbl
{
    BEGIN_INTERFACE

    HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
        __RPC__in __FIVectorView_1_float * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);

    ULONG ( STDMETHODCALLTYPE *AddRef )( __RPC__in __FIVectorView_1_float * This);

    ULONG ( STDMETHODCALLTYPE *Release )( __RPC__in __FIVectorView_1_float * This);

    HRESULT ( STDMETHODCALLTYPE *GetIids )( __RPC__in __FIVectorView_1_float * This,
                                            /* [out] */ __RPC__out ULONG *iidCount,
                                            /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);

    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )( 
        __RPC__in __FIVectorView_1_float * This,
            /* [out] */ __RPC__deref_out_opt HSTRING *className);

    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )( 
        __RPC__in __FIVectorView_1_float * This,
            /* [out] */ __RPC__out TrustLevel *trustLevel);

    HRESULT ( STDMETHODCALLTYPE *GetAt )( 
                                         __RPC__in __FIVectorView_1_float * This,
                                         /* [in] */ unsigned int index,
                                         /* [retval][out] */ __RPC__out float *item);

        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Size )( 
            __RPC__in __FIVectorView_1_float * This,
            /* [retval][out] */ __RPC__out unsigned int *size);

        HRESULT ( STDMETHODCALLTYPE *IndexOf )( 
                                               __RPC__in __FIVectorView_1_float * This,
            /* [in] */ float item,
            /* [out] */ __RPC__out unsigned int *index,
            /* [retval][out] */ __RPC__out boolean *found);

        HRESULT ( STDMETHODCALLTYPE *GetMany )( 
                                               __RPC__in __FIVectorView_1_float * This,
            /* [in] */ unsigned int startIndex,
            /* [in] */ unsigned int capacity,
            /* [size_is][length_is][out] */ __RPC__out_ecount_part(capacity, *actual) float *items,
            /* [retval][out] */ __RPC__out unsigned int *actual);

        END_INTERFACE
} __FIVectorView_1_floatVtbl;

interface __FIVectorView_1_float
{
    CONST_VTBL struct __FIVectorView_1_floatVtbl *lpVtbl;
};



#ifdef COBJMACROS


#define __FIVectorView_1_float_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define __FIVectorView_1_float_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define __FIVectorView_1_float_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define __FIVectorView_1_float_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 

#define __FIVectorView_1_float_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 

#define __FIVectorView_1_float_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 


#define __FIVectorView_1_float_GetAt(This,index,item)	\
    ( (This)->lpVtbl -> GetAt(This,index,item) ) 

#define __FIVectorView_1_float_get_Size(This,size)	\
    ( (This)->lpVtbl -> get_Size(This,size) ) 

#define __FIVectorView_1_float_IndexOf(This,item,index,found)	\
    ( (This)->lpVtbl -> IndexOf(This,item,index,found) ) 

#define __FIVectorView_1_float_GetMany(This,startIndex,capacity,items,actual)	\
    ( (This)->lpVtbl -> GetMany(This,startIndex,capacity,items,actual) ) 

#endif /* COBJMACROS */



#endif // ____FIVectorView_1_float_INTERFACE_DEFINED__


#if !defined(____FIIterator_1_byte_INTERFACE_DEFINED__)
#define ____FIIterator_1_byte_INTERFACE_DEFINED__

typedef interface __FIIterator_1_byte __FIIterator_1_byte;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIIterator_1_byte;

typedef struct __FIIterator_1_byteVtbl
{
    BEGIN_INTERFACE

    HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
        __RPC__in __FIIterator_1_byte * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);
    ULONG ( STDMETHODCALLTYPE *AddRef )(__RPC__in __FIIterator_1_byte * This);
    ULONG ( STDMETHODCALLTYPE *Release )(__RPC__in __FIIterator_1_byte * This);
    HRESULT ( STDMETHODCALLTYPE *GetIids )(__RPC__in __FIIterator_1_byte * This,
        /* [out] */ __RPC__out ULONG *iidCount,
        /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);

    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(__RPC__in __FIIterator_1_byte * This, /* [out] */ __RPC__deref_out_opt HSTRING *className);
    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(__RPC__in __FIIterator_1_byte * This, /* [out] */ __RPC__out TrustLevel *trustLevel);

    /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Current )(__RPC__in __FIIterator_1_byte * This, /* [retval][out] */ __RPC__out byte *current);
    /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_HasCurrent )(__RPC__in __FIIterator_1_byte * This, /* [retval][out] */ __RPC__out boolean *hasCurrent);
    HRESULT ( STDMETHODCALLTYPE *MoveNext )(__RPC__in __FIIterator_1_byte * This, /* [retval][out] */ __RPC__out boolean *hasCurrent);
    HRESULT ( STDMETHODCALLTYPE *GetMany )(__RPC__in __FIIterator_1_byte * This,
        /* [in] */ unsigned int capacity,
        /* [size_is][length_is][out] */ __RPC__out_ecount_part(capacity, *actual) byte *items,
        /* [retval][out] */ __RPC__out unsigned int *actual);

    END_INTERFACE
} __FIIterator_1_byteVtbl;

interface __FIIterator_1_byte
{
    CONST_VTBL struct __FIIterator_1_byteVtbl *lpVtbl;
};



#ifdef COBJMACROS


#define __FIIterator_1_byte_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define __FIIterator_1_byte_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define __FIIterator_1_byte_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define __FIIterator_1_byte_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 

#define __FIIterator_1_byte_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 

#define __FIIterator_1_byte_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 


#define __FIIterator_1_byte_get_Current(This,current)	\
    ( (This)->lpVtbl -> get_Current(This,current) ) 

#define __FIIterator_1_byte_get_HasCurrent(This,hasCurrent)	\
    ( (This)->lpVtbl -> get_HasCurrent(This,hasCurrent) ) 

#define __FIIterator_1_byte_MoveNext(This,hasCurrent)	\
    ( (This)->lpVtbl -> MoveNext(This,hasCurrent) ) 

#define __FIIterator_1_byte_GetMany(This,capacity,items,actual)	\
    ( (This)->lpVtbl -> GetMany(This,capacity,items,actual) ) 

#endif /* COBJMACROS */


#endif // ____FIIterator_1_byte_INTERFACE_DEFINED__


#if !defined(____FIIterable_1_byte_INTERFACE_DEFINED__)
#define ____FIIterable_1_byte_INTERFACE_DEFINED__

typedef interface __FIIterable_1_byte __FIIterable_1_byte;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIIterable_1_byte;

typedef  struct __FIIterable_1_byteVtbl
{
    BEGIN_INTERFACE

    HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
        __RPC__in __FIIterable_1_byte * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);

    ULONG ( STDMETHODCALLTYPE *AddRef )(__RPC__in __FIIterable_1_byte * This);

    ULONG ( STDMETHODCALLTYPE *Release )(__RPC__in __FIIterable_1_byte * This);

    HRESULT ( STDMETHODCALLTYPE *GetIids )(__RPC__in __FIIterable_1_byte * This,
                                           /* [out] */ __RPC__out ULONG *iidCount,
                                           /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);

    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(__RPC__in __FIIterable_1_byte * This, /* [out] */ __RPC__deref_out_opt HSTRING *className);

    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(__RPC__in __FIIterable_1_byte * This, /* [out] */ __RPC__out TrustLevel *trustLevel);

    HRESULT ( STDMETHODCALLTYPE *First )(__RPC__in __FIIterable_1_byte * This, /* [retval][out] */ __RPC__deref_out_opt __FIIterator_1_byte **first);

    END_INTERFACE
} __FIIterable_1_byteVtbl;

interface __FIIterable_1_byte
{
    CONST_VTBL struct __FIIterable_1_byteVtbl *lpVtbl;
};

#ifdef COBJMACROS

#define __FIIterable_1_byte_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define __FIIterable_1_byte_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define __FIIterable_1_byte_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define __FIIterable_1_byte_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 

#define __FIIterable_1_byte_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 

#define __FIIterable_1_byte_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 


#define __FIIterable_1_byte_First(This,first)	\
    ( (This)->lpVtbl -> First(This,first) ) 

#endif /* COBJMACROS */


#endif // ____FIIterable_1_byte_INTERFACE_DEFINED__


#if !defined(____FIVectorView_1_byte_INTERFACE_DEFINED__)
#define ____FIVectorView_1_byte_INTERFACE_DEFINED__

typedef interface __FIVectorView_1_byte __FIVectorView_1_byte;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIVectorView_1_byte;

typedef struct __FIVectorView_1_byteVtbl
{
    BEGIN_INTERFACE

    HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
        __RPC__in __FIVectorView_1_byte * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);

    ULONG ( STDMETHODCALLTYPE *AddRef )( __RPC__in __FIVectorView_1_byte * This);

    ULONG ( STDMETHODCALLTYPE *Release )( __RPC__in __FIVectorView_1_byte * This);

    HRESULT ( STDMETHODCALLTYPE *GetIids )( __RPC__in __FIVectorView_1_byte * This,
                                            /* [out] */ __RPC__out ULONG *iidCount,
                                            /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);

    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )( 
        __RPC__in __FIVectorView_1_byte * This,
            /* [out] */ __RPC__deref_out_opt HSTRING *className);

    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )( 
        __RPC__in __FIVectorView_1_byte * This,
            /* [out] */ __RPC__out TrustLevel *trustLevel);

    HRESULT ( STDMETHODCALLTYPE *GetAt )( 
                                         __RPC__in __FIVectorView_1_byte * This,
                                         /* [in] */ unsigned int index,
                                         /* [retval][out] */ __RPC__out byte *item);

        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Size )( 
            __RPC__in __FIVectorView_1_byte * This,
            /* [retval][out] */ __RPC__out unsigned int *size);

        HRESULT ( STDMETHODCALLTYPE *IndexOf )( 
                                               __RPC__in __FIVectorView_1_byte * This,
            /* [in] */ byte item,
            /* [out] */ __RPC__out unsigned int *index,
            /* [retval][out] */ __RPC__out boolean *found);

        HRESULT ( STDMETHODCALLTYPE *GetMany )( 
                                               __RPC__in __FIVectorView_1_byte * This,
            /* [in] */ unsigned int startIndex,
            /* [in] */ unsigned int capacity,
            /* [size_is][length_is][out] */ __RPC__out_ecount_part(capacity, *actual) byte *items,
            /* [retval][out] */ __RPC__out unsigned int *actual);

        END_INTERFACE
} __FIVectorView_1_byteVtbl;

interface __FIVectorView_1_byte
{
    CONST_VTBL struct __FIVectorView_1_byteVtbl *lpVtbl;
};



#ifdef COBJMACROS


#define __FIVectorView_1_byte_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define __FIVectorView_1_byte_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define __FIVectorView_1_byte_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define __FIVectorView_1_byte_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 

#define __FIVectorView_1_byte_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 

#define __FIVectorView_1_byte_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 


#define __FIVectorView_1_byte_GetAt(This,index,item)	\
    ( (This)->lpVtbl -> GetAt(This,index,item) ) 

#define __FIVectorView_1_byte_get_Size(This,size)	\
    ( (This)->lpVtbl -> get_Size(This,size) ) 

#define __FIVectorView_1_byte_IndexOf(This,item,index,found)	\
    ( (This)->lpVtbl -> IndexOf(This,item,index,found) ) 

#define __FIVectorView_1_byte_GetMany(This,startIndex,capacity,items,actual)	\
    ( (This)->lpVtbl -> GetMany(This,startIndex,capacity,items,actual) ) 

#endif /* COBJMACROS */



#endif // ____FIVectorView_1_byte_INTERFACE_DEFINED__


#if !defined(____FIIterator_1_UINT16_INTERFACE_DEFINED__)
#define ____FIIterator_1_UINT16_INTERFACE_DEFINED__

typedef interface __FIIterator_1_UINT16 __FIIterator_1_UINT16;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIIterator_1_UINT16;

typedef struct __FIIterator_1_UINT16Vtbl
{
    BEGIN_INTERFACE

    HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
        __RPC__in __FIIterator_1_UINT16 * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);
    ULONG ( STDMETHODCALLTYPE *AddRef )(__RPC__in __FIIterator_1_UINT16 * This);
    ULONG ( STDMETHODCALLTYPE *Release )(__RPC__in __FIIterator_1_UINT16 * This);
    HRESULT ( STDMETHODCALLTYPE *GetIids )(__RPC__in __FIIterator_1_UINT16 * This,
        /* [out] */ __RPC__out ULONG *iidCount,
        /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);

    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(__RPC__in __FIIterator_1_UINT16 * This, /* [out] */ __RPC__deref_out_opt HSTRING *className);
    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(__RPC__in __FIIterator_1_UINT16 * This, /* [out] */ __RPC__out TrustLevel *trustLevel);

    /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Current )(__RPC__in __FIIterator_1_UINT16 * This, /* [retval][out] */ __RPC__out unsigned short *current);
    /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_HasCurrent )(__RPC__in __FIIterator_1_UINT16 * This, /* [retval][out] */ __RPC__out boolean *hasCurrent);
    HRESULT ( STDMETHODCALLTYPE *MoveNext )(__RPC__in __FIIterator_1_UINT16 * This, /* [retval][out] */ __RPC__out boolean *hasCurrent);
    HRESULT ( STDMETHODCALLTYPE *GetMany )(__RPC__in __FIIterator_1_UINT16 * This,
        /* [in] */ unsigned int capacity,
        /* [size_is][length_is][out] */ __RPC__out_ecount_part(capacity, *actual) unsigned short *items,
        /* [retval][out] */ __RPC__out unsigned int *actual);

    END_INTERFACE
} __FIIterator_1_UINT16Vtbl;

interface __FIIterator_1_UINT16
{
    CONST_VTBL struct __FIIterator_1_UINT16Vtbl *lpVtbl;
};



#ifdef COBJMACROS


#define __FIIterator_1_UINT16_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define __FIIterator_1_UINT16_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define __FIIterator_1_UINT16_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define __FIIterator_1_UINT16_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 

#define __FIIterator_1_UINT16_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 

#define __FIIterator_1_UINT16_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 


#define __FIIterator_1_UINT16_get_Current(This,current)	\
    ( (This)->lpVtbl -> get_Current(This,current) ) 

#define __FIIterator_1_UINT16_get_HasCurrent(This,hasCurrent)	\
    ( (This)->lpVtbl -> get_HasCurrent(This,hasCurrent) ) 

#define __FIIterator_1_UINT16_MoveNext(This,hasCurrent)	\
    ( (This)->lpVtbl -> MoveNext(This,hasCurrent) ) 

#define __FIIterator_1_UINT16_GetMany(This,capacity,items,actual)	\
    ( (This)->lpVtbl -> GetMany(This,capacity,items,actual) ) 

#endif /* COBJMACROS */


#endif // ____FIIterator_1_UINT16_INTERFACE_DEFINED__


#if !defined(____FIIterable_1_UINT16_INTERFACE_DEFINED__)
#define ____FIIterable_1_UINT16_INTERFACE_DEFINED__

typedef interface __FIIterable_1_UINT16 __FIIterable_1_UINT16;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIIterable_1_UINT16;

typedef  struct __FIIterable_1_UINT16Vtbl
{
    BEGIN_INTERFACE

    HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
        __RPC__in __FIIterable_1_UINT16 * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);

    ULONG ( STDMETHODCALLTYPE *AddRef )(__RPC__in __FIIterable_1_UINT16 * This);

    ULONG ( STDMETHODCALLTYPE *Release )(__RPC__in __FIIterable_1_UINT16 * This);

    HRESULT ( STDMETHODCALLTYPE *GetIids )(__RPC__in __FIIterable_1_UINT16 * This,
                                           /* [out] */ __RPC__out ULONG *iidCount,
                                           /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);

    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(__RPC__in __FIIterable_1_UINT16 * This, /* [out] */ __RPC__deref_out_opt HSTRING *className);

    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(__RPC__in __FIIterable_1_UINT16 * This, /* [out] */ __RPC__out TrustLevel *trustLevel);

    HRESULT ( STDMETHODCALLTYPE *First )(__RPC__in __FIIterable_1_UINT16 * This, /* [retval][out] */ __RPC__deref_out_opt __FIIterator_1_UINT16 **first);

    END_INTERFACE
} __FIIterable_1_UINT16Vtbl;

interface __FIIterable_1_UINT16
{
    CONST_VTBL struct __FIIterable_1_UINT16Vtbl *lpVtbl;
};

#ifdef COBJMACROS

#define __FIIterable_1_UINT16_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define __FIIterable_1_UINT16_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define __FIIterable_1_UINT16_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define __FIIterable_1_UINT16_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 

#define __FIIterable_1_UINT16_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 

#define __FIIterable_1_UINT16_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 


#define __FIIterable_1_UINT16_First(This,first)	\
    ( (This)->lpVtbl -> First(This,first) ) 

#endif /* COBJMACROS */


#endif // ____FIIterable_1_UINT16_INTERFACE_DEFINED__


#if !defined(____FIVectorView_1_UINT16_INTERFACE_DEFINED__)
#define ____FIVectorView_1_UINT16_INTERFACE_DEFINED__

typedef interface __FIVectorView_1_UINT16 __FIVectorView_1_UINT16;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIVectorView_1_UINT16;

typedef struct __FIVectorView_1_UINT16Vtbl
{
    BEGIN_INTERFACE

    HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
        __RPC__in __FIVectorView_1_UINT16 * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);

    ULONG ( STDMETHODCALLTYPE *AddRef )( __RPC__in __FIVectorView_1_UINT16 * This);

    ULONG ( STDMETHODCALLTYPE *Release )( __RPC__in __FIVectorView_1_UINT16 * This);

    HRESULT ( STDMETHODCALLTYPE *GetIids )( __RPC__in __FIVectorView_1_UINT16 * This,
                                            /* [out] */ __RPC__out ULONG *iidCount,
                                            /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);

    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )( 
        __RPC__in __FIVectorView_1_UINT16 * This,
            /* [out] */ __RPC__deref_out_opt HSTRING *className);

    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )( 
        __RPC__in __FIVectorView_1_UINT16 * This,
            /* [out] */ __RPC__out TrustLevel *trustLevel);

    HRESULT ( STDMETHODCALLTYPE *GetAt )( 
                                         __RPC__in __FIVectorView_1_UINT16 * This,
                                         /* [in] */ unsigned int index,
                                         /* [retval][out] */ __RPC__out unsigned short *item);

        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Size )( 
            __RPC__in __FIVectorView_1_UINT16 * This,
            /* [retval][out] */ __RPC__out unsigned int *size);

        HRESULT ( STDMETHODCALLTYPE *IndexOf )( 
                                               __RPC__in __FIVectorView_1_UINT16 * This,
            /* [in] */ unsigned short item,
            /* [out] */ __RPC__out unsigned int *index,
            /* [retval][out] */ __RPC__out boolean *found);

        HRESULT ( STDMETHODCALLTYPE *GetMany )( 
                                               __RPC__in __FIVectorView_1_UINT16 * This,
            /* [in] */ unsigned int startIndex,
            /* [in] */ unsigned int capacity,
            /* [size_is][length_is][out] */ __RPC__out_ecount_part(capacity, *actual) unsigned short *items,
            /* [retval][out] */ __RPC__out unsigned int *actual);

        END_INTERFACE
} __FIVectorView_1_UINT16Vtbl;

interface __FIVectorView_1_UINT16
{
    CONST_VTBL struct __FIVectorView_1_UINT16Vtbl *lpVtbl;
};



#ifdef COBJMACROS


#define __FIVectorView_1_UINT16_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define __FIVectorView_1_UINT16_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define __FIVectorView_1_UINT16_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define __FIVectorView_1_UINT16_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 

#define __FIVectorView_1_UINT16_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 

#define __FIVectorView_1_UINT16_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 


#define __FIVectorView_1_UINT16_GetAt(This,index,item)	\
    ( (This)->lpVtbl -> GetAt(This,index,item) ) 

#define __FIVectorView_1_UINT16_get_Size(This,size)	\
    ( (This)->lpVtbl -> get_Size(This,size) ) 

#define __FIVectorView_1_UINT16_IndexOf(This,item,index,found)	\
    ( (This)->lpVtbl -> IndexOf(This,item,index,found) ) 

#define __FIVectorView_1_UINT16_GetMany(This,startIndex,capacity,items,actual)	\
    ( (This)->lpVtbl -> GetMany(This,startIndex,capacity,items,actual) ) 

#endif /* COBJMACROS */



#endif // ____FIVectorView_1_UINT16_INTERFACE_DEFINED__


#if !defined(____FIIterator_1_short_INTERFACE_DEFINED__)
#define ____FIIterator_1_short_INTERFACE_DEFINED__

typedef interface __FIIterator_1_short __FIIterator_1_short;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIIterator_1_short;

typedef struct __FIIterator_1_shortVtbl
{
    BEGIN_INTERFACE

    HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
        __RPC__in __FIIterator_1_short * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);
    ULONG ( STDMETHODCALLTYPE *AddRef )(__RPC__in __FIIterator_1_short * This);
    ULONG ( STDMETHODCALLTYPE *Release )(__RPC__in __FIIterator_1_short * This);
    HRESULT ( STDMETHODCALLTYPE *GetIids )(__RPC__in __FIIterator_1_short * This,
        /* [out] */ __RPC__out ULONG *iidCount,
        /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);

    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(__RPC__in __FIIterator_1_short * This, /* [out] */ __RPC__deref_out_opt HSTRING *className);
    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(__RPC__in __FIIterator_1_short * This, /* [out] */ __RPC__out TrustLevel *trustLevel);

    /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Current )(__RPC__in __FIIterator_1_short * This, /* [retval][out] */ __RPC__out short *current);
    /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_HasCurrent )(__RPC__in __FIIterator_1_short * This, /* [retval][out] */ __RPC__out boolean *hasCurrent);
    HRESULT ( STDMETHODCALLTYPE *MoveNext )(__RPC__in __FIIterator_1_short * This, /* [retval][out] */ __RPC__out boolean *hasCurrent);
    HRESULT ( STDMETHODCALLTYPE *GetMany )(__RPC__in __FIIterator_1_short * This,
        /* [in] */ unsigned int capacity,
        /* [size_is][length_is][out] */ __RPC__out_ecount_part(capacity, *actual) short *items,
        /* [retval][out] */ __RPC__out unsigned int *actual);

    END_INTERFACE
} __FIIterator_1_shortVtbl;

interface __FIIterator_1_short
{
    CONST_VTBL struct __FIIterator_1_shortVtbl *lpVtbl;
};



#ifdef COBJMACROS


#define __FIIterator_1_short_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define __FIIterator_1_short_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define __FIIterator_1_short_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define __FIIterator_1_short_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 

#define __FIIterator_1_short_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 

#define __FIIterator_1_short_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 


#define __FIIterator_1_short_get_Current(This,current)	\
    ( (This)->lpVtbl -> get_Current(This,current) ) 

#define __FIIterator_1_short_get_HasCurrent(This,hasCurrent)	\
    ( (This)->lpVtbl -> get_HasCurrent(This,hasCurrent) ) 

#define __FIIterator_1_short_MoveNext(This,hasCurrent)	\
    ( (This)->lpVtbl -> MoveNext(This,hasCurrent) ) 

#define __FIIterator_1_short_GetMany(This,capacity,items,actual)	\
    ( (This)->lpVtbl -> GetMany(This,capacity,items,actual) ) 

#endif /* COBJMACROS */


#endif // ____FIIterator_1_short_INTERFACE_DEFINED__


#if !defined(____FIIterable_1_short_INTERFACE_DEFINED__)
#define ____FIIterable_1_short_INTERFACE_DEFINED__

typedef interface __FIIterable_1_short __FIIterable_1_short;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIIterable_1_short;

typedef  struct __FIIterable_1_shortVtbl
{
    BEGIN_INTERFACE

    HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
        __RPC__in __FIIterable_1_short * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);

    ULONG ( STDMETHODCALLTYPE *AddRef )(__RPC__in __FIIterable_1_short * This);

    ULONG ( STDMETHODCALLTYPE *Release )(__RPC__in __FIIterable_1_short * This);

    HRESULT ( STDMETHODCALLTYPE *GetIids )(__RPC__in __FIIterable_1_short * This,
                                           /* [out] */ __RPC__out ULONG *iidCount,
                                           /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);

    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(__RPC__in __FIIterable_1_short * This, /* [out] */ __RPC__deref_out_opt HSTRING *className);

    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(__RPC__in __FIIterable_1_short * This, /* [out] */ __RPC__out TrustLevel *trustLevel);

    HRESULT ( STDMETHODCALLTYPE *First )(__RPC__in __FIIterable_1_short * This, /* [retval][out] */ __RPC__deref_out_opt __FIIterator_1_short **first);

    END_INTERFACE
} __FIIterable_1_shortVtbl;

interface __FIIterable_1_short
{
    CONST_VTBL struct __FIIterable_1_shortVtbl *lpVtbl;
};

#ifdef COBJMACROS

#define __FIIterable_1_short_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define __FIIterable_1_short_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define __FIIterable_1_short_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define __FIIterable_1_short_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 

#define __FIIterable_1_short_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 

#define __FIIterable_1_short_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 


#define __FIIterable_1_short_First(This,first)	\
    ( (This)->lpVtbl -> First(This,first) ) 

#endif /* COBJMACROS */


#endif // ____FIIterable_1_short_INTERFACE_DEFINED__


#if !defined(____FIVectorView_1_short_INTERFACE_DEFINED__)
#define ____FIVectorView_1_short_INTERFACE_DEFINED__

typedef interface __FIVectorView_1_short __FIVectorView_1_short;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIVectorView_1_short;

typedef struct __FIVectorView_1_shortVtbl
{
    BEGIN_INTERFACE

    HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
        __RPC__in __FIVectorView_1_short * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);

    ULONG ( STDMETHODCALLTYPE *AddRef )( __RPC__in __FIVectorView_1_short * This);

    ULONG ( STDMETHODCALLTYPE *Release )( __RPC__in __FIVectorView_1_short * This);

    HRESULT ( STDMETHODCALLTYPE *GetIids )( __RPC__in __FIVectorView_1_short * This,
                                            /* [out] */ __RPC__out ULONG *iidCount,
                                            /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);

    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )( 
        __RPC__in __FIVectorView_1_short * This,
            /* [out] */ __RPC__deref_out_opt HSTRING *className);

    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )( 
        __RPC__in __FIVectorView_1_short * This,
            /* [out] */ __RPC__out TrustLevel *trustLevel);

    HRESULT ( STDMETHODCALLTYPE *GetAt )( 
                                         __RPC__in __FIVectorView_1_short * This,
                                         /* [in] */ unsigned int index,
                                         /* [retval][out] */ __RPC__out short *item);

        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Size )( 
            __RPC__in __FIVectorView_1_short * This,
            /* [retval][out] */ __RPC__out unsigned int *size);

        HRESULT ( STDMETHODCALLTYPE *IndexOf )( 
                                               __RPC__in __FIVectorView_1_short * This,
            /* [in] */ short item,
            /* [out] */ __RPC__out unsigned int *index,
            /* [retval][out] */ __RPC__out boolean *found);

        HRESULT ( STDMETHODCALLTYPE *GetMany )( 
                                               __RPC__in __FIVectorView_1_short * This,
            /* [in] */ unsigned int startIndex,
            /* [in] */ unsigned int capacity,
            /* [size_is][length_is][out] */ __RPC__out_ecount_part(capacity, *actual) short *items,
            /* [retval][out] */ __RPC__out unsigned int *actual);

        END_INTERFACE
} __FIVectorView_1_shortVtbl;

interface __FIVectorView_1_short
{
    CONST_VTBL struct __FIVectorView_1_shortVtbl *lpVtbl;
};



#ifdef COBJMACROS


#define __FIVectorView_1_short_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define __FIVectorView_1_short_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define __FIVectorView_1_short_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define __FIVectorView_1_short_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 

#define __FIVectorView_1_short_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 

#define __FIVectorView_1_short_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 


#define __FIVectorView_1_short_GetAt(This,index,item)	\
    ( (This)->lpVtbl -> GetAt(This,index,item) ) 

#define __FIVectorView_1_short_get_Size(This,size)	\
    ( (This)->lpVtbl -> get_Size(This,size) ) 

#define __FIVectorView_1_short_IndexOf(This,item,index,found)	\
    ( (This)->lpVtbl -> IndexOf(This,item,index,found) ) 

#define __FIVectorView_1_short_GetMany(This,startIndex,capacity,items,actual)	\
    ( (This)->lpVtbl -> GetMany(This,startIndex,capacity,items,actual) ) 

#endif /* COBJMACROS */



#endif // ____FIVectorView_1_short_INTERFACE_DEFINED__


#if !defined(____FIIterator_1_UINT32_INTERFACE_DEFINED__)
#define ____FIIterator_1_UINT32_INTERFACE_DEFINED__

typedef interface __FIIterator_1_UINT32 __FIIterator_1_UINT32;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIIterator_1_UINT32;

typedef struct __FIIterator_1_UINT32Vtbl
{
    BEGIN_INTERFACE

    HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
        __RPC__in __FIIterator_1_UINT32 * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);
    ULONG ( STDMETHODCALLTYPE *AddRef )(__RPC__in __FIIterator_1_UINT32 * This);
    ULONG ( STDMETHODCALLTYPE *Release )(__RPC__in __FIIterator_1_UINT32 * This);
    HRESULT ( STDMETHODCALLTYPE *GetIids )(__RPC__in __FIIterator_1_UINT32 * This,
        /* [out] */ __RPC__out ULONG *iidCount,
        /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);

    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(__RPC__in __FIIterator_1_UINT32 * This, /* [out] */ __RPC__deref_out_opt HSTRING *className);
    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(__RPC__in __FIIterator_1_UINT32 * This, /* [out] */ __RPC__out TrustLevel *trustLevel);

    /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Current )(__RPC__in __FIIterator_1_UINT32 * This, /* [retval][out] */ __RPC__out unsigned int *current);
    /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_HasCurrent )(__RPC__in __FIIterator_1_UINT32 * This, /* [retval][out] */ __RPC__out boolean *hasCurrent);
    HRESULT ( STDMETHODCALLTYPE *MoveNext )(__RPC__in __FIIterator_1_UINT32 * This, /* [retval][out] */ __RPC__out boolean *hasCurrent);
    HRESULT ( STDMETHODCALLTYPE *GetMany )(__RPC__in __FIIterator_1_UINT32 * This,
        /* [in] */ unsigned int capacity,
        /* [size_is][length_is][out] */ __RPC__out_ecount_part(capacity, *actual) unsigned int *items,
        /* [retval][out] */ __RPC__out unsigned int *actual);

    END_INTERFACE
} __FIIterator_1_UINT32Vtbl;

interface __FIIterator_1_UINT32
{
    CONST_VTBL struct __FIIterator_1_UINT32Vtbl *lpVtbl;
};



#ifdef COBJMACROS


#define __FIIterator_1_UINT32_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define __FIIterator_1_UINT32_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define __FIIterator_1_UINT32_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define __FIIterator_1_UINT32_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 

#define __FIIterator_1_UINT32_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 

#define __FIIterator_1_UINT32_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 


#define __FIIterator_1_UINT32_get_Current(This,current)	\
    ( (This)->lpVtbl -> get_Current(This,current) ) 

#define __FIIterator_1_UINT32_get_HasCurrent(This,hasCurrent)	\
    ( (This)->lpVtbl -> get_HasCurrent(This,hasCurrent) ) 

#define __FIIterator_1_UINT32_MoveNext(This,hasCurrent)	\
    ( (This)->lpVtbl -> MoveNext(This,hasCurrent) ) 

#define __FIIterator_1_UINT32_GetMany(This,capacity,items,actual)	\
    ( (This)->lpVtbl -> GetMany(This,capacity,items,actual) ) 

#endif /* COBJMACROS */


#endif // ____FIIterator_1_UINT32_INTERFACE_DEFINED__


#if !defined(____FIIterable_1_UINT32_INTERFACE_DEFINED__)
#define ____FIIterable_1_UINT32_INTERFACE_DEFINED__

typedef interface __FIIterable_1_UINT32 __FIIterable_1_UINT32;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIIterable_1_UINT32;

typedef  struct __FIIterable_1_UINT32Vtbl
{
    BEGIN_INTERFACE

    HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
        __RPC__in __FIIterable_1_UINT32 * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);

    ULONG ( STDMETHODCALLTYPE *AddRef )(__RPC__in __FIIterable_1_UINT32 * This);

    ULONG ( STDMETHODCALLTYPE *Release )(__RPC__in __FIIterable_1_UINT32 * This);

    HRESULT ( STDMETHODCALLTYPE *GetIids )(__RPC__in __FIIterable_1_UINT32 * This,
                                           /* [out] */ __RPC__out ULONG *iidCount,
                                           /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);

    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(__RPC__in __FIIterable_1_UINT32 * This, /* [out] */ __RPC__deref_out_opt HSTRING *className);

    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(__RPC__in __FIIterable_1_UINT32 * This, /* [out] */ __RPC__out TrustLevel *trustLevel);

    HRESULT ( STDMETHODCALLTYPE *First )(__RPC__in __FIIterable_1_UINT32 * This, /* [retval][out] */ __RPC__deref_out_opt __FIIterator_1_UINT32 **first);

    END_INTERFACE
} __FIIterable_1_UINT32Vtbl;

interface __FIIterable_1_UINT32
{
    CONST_VTBL struct __FIIterable_1_UINT32Vtbl *lpVtbl;
};

#ifdef COBJMACROS

#define __FIIterable_1_UINT32_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define __FIIterable_1_UINT32_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define __FIIterable_1_UINT32_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define __FIIterable_1_UINT32_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 

#define __FIIterable_1_UINT32_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 

#define __FIIterable_1_UINT32_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 


#define __FIIterable_1_UINT32_First(This,first)	\
    ( (This)->lpVtbl -> First(This,first) ) 

#endif /* COBJMACROS */


#endif // ____FIIterable_1_UINT32_INTERFACE_DEFINED__


#if !defined(____FIVectorView_1_UINT32_INTERFACE_DEFINED__)
#define ____FIVectorView_1_UINT32_INTERFACE_DEFINED__

typedef interface __FIVectorView_1_UINT32 __FIVectorView_1_UINT32;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIVectorView_1_UINT32;

typedef struct __FIVectorView_1_UINT32Vtbl
{
    BEGIN_INTERFACE

    HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
        __RPC__in __FIVectorView_1_UINT32 * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);

    ULONG ( STDMETHODCALLTYPE *AddRef )( __RPC__in __FIVectorView_1_UINT32 * This);

    ULONG ( STDMETHODCALLTYPE *Release )( __RPC__in __FIVectorView_1_UINT32 * This);

    HRESULT ( STDMETHODCALLTYPE *GetIids )( __RPC__in __FIVectorView_1_UINT32 * This,
                                            /* [out] */ __RPC__out ULONG *iidCount,
                                            /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);

    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )( 
        __RPC__in __FIVectorView_1_UINT32 * This,
            /* [out] */ __RPC__deref_out_opt HSTRING *className);

    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )( 
        __RPC__in __FIVectorView_1_UINT32 * This,
            /* [out] */ __RPC__out TrustLevel *trustLevel);

    HRESULT ( STDMETHODCALLTYPE *GetAt )( 
                                         __RPC__in __FIVectorView_1_UINT32 * This,
                                         /* [in] */ unsigned int index,
                                         /* [retval][out] */ __RPC__out unsigned int *item);

        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Size )( 
            __RPC__in __FIVectorView_1_UINT32 * This,
            /* [retval][out] */ __RPC__out unsigned int *size);

        HRESULT ( STDMETHODCALLTYPE *IndexOf )( 
                                               __RPC__in __FIVectorView_1_UINT32 * This,
            /* [in] */ unsigned int item,
            /* [out] */ __RPC__out unsigned int *index,
            /* [retval][out] */ __RPC__out boolean *found);

        HRESULT ( STDMETHODCALLTYPE *GetMany )( 
                                               __RPC__in __FIVectorView_1_UINT32 * This,
            /* [in] */ unsigned int startIndex,
            /* [in] */ unsigned int capacity,
            /* [size_is][length_is][out] */ __RPC__out_ecount_part(capacity, *actual) unsigned int *items,
            /* [retval][out] */ __RPC__out unsigned int *actual);

        END_INTERFACE
} __FIVectorView_1_UINT32Vtbl;

interface __FIVectorView_1_UINT32
{
    CONST_VTBL struct __FIVectorView_1_UINT32Vtbl *lpVtbl;
};



#ifdef COBJMACROS


#define __FIVectorView_1_UINT32_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define __FIVectorView_1_UINT32_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define __FIVectorView_1_UINT32_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define __FIVectorView_1_UINT32_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 

#define __FIVectorView_1_UINT32_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 

#define __FIVectorView_1_UINT32_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 


#define __FIVectorView_1_UINT32_GetAt(This,index,item)	\
    ( (This)->lpVtbl -> GetAt(This,index,item) ) 

#define __FIVectorView_1_UINT32_get_Size(This,size)	\
    ( (This)->lpVtbl -> get_Size(This,size) ) 

#define __FIVectorView_1_UINT32_IndexOf(This,item,index,found)	\
    ( (This)->lpVtbl -> IndexOf(This,item,index,found) ) 

#define __FIVectorView_1_UINT32_GetMany(This,startIndex,capacity,items,actual)	\
    ( (This)->lpVtbl -> GetMany(This,startIndex,capacity,items,actual) ) 

#endif /* COBJMACROS */



#endif // ____FIVectorView_1_UINT32_INTERFACE_DEFINED__


#if !defined(____FIIterator_1_int_INTERFACE_DEFINED__)
#define ____FIIterator_1_int_INTERFACE_DEFINED__

typedef interface __FIIterator_1_int __FIIterator_1_int;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIIterator_1_int;

typedef struct __FIIterator_1_intVtbl
{
    BEGIN_INTERFACE

    HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
        __RPC__in __FIIterator_1_int * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);
    ULONG ( STDMETHODCALLTYPE *AddRef )(__RPC__in __FIIterator_1_int * This);
    ULONG ( STDMETHODCALLTYPE *Release )(__RPC__in __FIIterator_1_int * This);
    HRESULT ( STDMETHODCALLTYPE *GetIids )(__RPC__in __FIIterator_1_int * This,
        /* [out] */ __RPC__out ULONG *iidCount,
        /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);

    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(__RPC__in __FIIterator_1_int * This, /* [out] */ __RPC__deref_out_opt HSTRING *className);
    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(__RPC__in __FIIterator_1_int * This, /* [out] */ __RPC__out TrustLevel *trustLevel);

    /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Current )(__RPC__in __FIIterator_1_int * This, /* [retval][out] */ __RPC__out int *current);
    /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_HasCurrent )(__RPC__in __FIIterator_1_int * This, /* [retval][out] */ __RPC__out boolean *hasCurrent);
    HRESULT ( STDMETHODCALLTYPE *MoveNext )(__RPC__in __FIIterator_1_int * This, /* [retval][out] */ __RPC__out boolean *hasCurrent);
    HRESULT ( STDMETHODCALLTYPE *GetMany )(__RPC__in __FIIterator_1_int * This,
        /* [in] */ unsigned int capacity,
        /* [size_is][length_is][out] */ __RPC__out_ecount_part(capacity, *actual) int *items,
        /* [retval][out] */ __RPC__out unsigned int *actual);

    END_INTERFACE
} __FIIterator_1_intVtbl;

interface __FIIterator_1_int
{
    CONST_VTBL struct __FIIterator_1_intVtbl *lpVtbl;
};



#ifdef COBJMACROS


#define __FIIterator_1_int_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define __FIIterator_1_int_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define __FIIterator_1_int_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define __FIIterator_1_int_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 

#define __FIIterator_1_int_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 

#define __FIIterator_1_int_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 


#define __FIIterator_1_int_get_Current(This,current)	\
    ( (This)->lpVtbl -> get_Current(This,current) ) 

#define __FIIterator_1_int_get_HasCurrent(This,hasCurrent)	\
    ( (This)->lpVtbl -> get_HasCurrent(This,hasCurrent) ) 

#define __FIIterator_1_int_MoveNext(This,hasCurrent)	\
    ( (This)->lpVtbl -> MoveNext(This,hasCurrent) ) 

#define __FIIterator_1_int_GetMany(This,capacity,items,actual)	\
    ( (This)->lpVtbl -> GetMany(This,capacity,items,actual) ) 

#endif /* COBJMACROS */


#endif // ____FIIterator_1_int_INTERFACE_DEFINED__


#if !defined(____FIIterable_1_int_INTERFACE_DEFINED__)
#define ____FIIterable_1_int_INTERFACE_DEFINED__

typedef interface __FIIterable_1_int __FIIterable_1_int;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIIterable_1_int;

typedef  struct __FIIterable_1_intVtbl
{
    BEGIN_INTERFACE

    HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
        __RPC__in __FIIterable_1_int * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);

    ULONG ( STDMETHODCALLTYPE *AddRef )(__RPC__in __FIIterable_1_int * This);

    ULONG ( STDMETHODCALLTYPE *Release )(__RPC__in __FIIterable_1_int * This);

    HRESULT ( STDMETHODCALLTYPE *GetIids )(__RPC__in __FIIterable_1_int * This,
                                           /* [out] */ __RPC__out ULONG *iidCount,
                                           /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);

    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(__RPC__in __FIIterable_1_int * This, /* [out] */ __RPC__deref_out_opt HSTRING *className);

    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(__RPC__in __FIIterable_1_int * This, /* [out] */ __RPC__out TrustLevel *trustLevel);

    HRESULT ( STDMETHODCALLTYPE *First )(__RPC__in __FIIterable_1_int * This, /* [retval][out] */ __RPC__deref_out_opt __FIIterator_1_int **first);

    END_INTERFACE
} __FIIterable_1_intVtbl;

interface __FIIterable_1_int
{
    CONST_VTBL struct __FIIterable_1_intVtbl *lpVtbl;
};

#ifdef COBJMACROS

#define __FIIterable_1_int_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define __FIIterable_1_int_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define __FIIterable_1_int_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define __FIIterable_1_int_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 

#define __FIIterable_1_int_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 

#define __FIIterable_1_int_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 


#define __FIIterable_1_int_First(This,first)	\
    ( (This)->lpVtbl -> First(This,first) ) 

#endif /* COBJMACROS */


#endif // ____FIIterable_1_int_INTERFACE_DEFINED__


#if !defined(____FIVectorView_1_int_INTERFACE_DEFINED__)
#define ____FIVectorView_1_int_INTERFACE_DEFINED__

typedef interface __FIVectorView_1_int __FIVectorView_1_int;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIVectorView_1_int;

typedef struct __FIVectorView_1_intVtbl
{
    BEGIN_INTERFACE

    HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
        __RPC__in __FIVectorView_1_int * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);

    ULONG ( STDMETHODCALLTYPE *AddRef )( __RPC__in __FIVectorView_1_int * This);

    ULONG ( STDMETHODCALLTYPE *Release )( __RPC__in __FIVectorView_1_int * This);

    HRESULT ( STDMETHODCALLTYPE *GetIids )( __RPC__in __FIVectorView_1_int * This,
                                            /* [out] */ __RPC__out ULONG *iidCount,
                                            /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);

    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )( 
        __RPC__in __FIVectorView_1_int * This,
            /* [out] */ __RPC__deref_out_opt HSTRING *className);

    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )( 
        __RPC__in __FIVectorView_1_int * This,
            /* [out] */ __RPC__out TrustLevel *trustLevel);

    HRESULT ( STDMETHODCALLTYPE *GetAt )( 
                                         __RPC__in __FIVectorView_1_int * This,
                                         /* [in] */ unsigned int index,
                                         /* [retval][out] */ __RPC__out int *item);

        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Size )( 
            __RPC__in __FIVectorView_1_int * This,
            /* [retval][out] */ __RPC__out unsigned int *size);

        HRESULT ( STDMETHODCALLTYPE *IndexOf )( 
                                               __RPC__in __FIVectorView_1_int * This,
            /* [in] */ int item,
            /* [out] */ __RPC__out unsigned int *index,
            /* [retval][out] */ __RPC__out boolean *found);

        HRESULT ( STDMETHODCALLTYPE *GetMany )( 
                                               __RPC__in __FIVectorView_1_int * This,
            /* [in] */ unsigned int startIndex,
            /* [in] */ unsigned int capacity,
            /* [size_is][length_is][out] */ __RPC__out_ecount_part(capacity, *actual) int *items,
            /* [retval][out] */ __RPC__out unsigned int *actual);

        END_INTERFACE
} __FIVectorView_1_intVtbl;

interface __FIVectorView_1_int
{
    CONST_VTBL struct __FIVectorView_1_intVtbl *lpVtbl;
};



#ifdef COBJMACROS


#define __FIVectorView_1_int_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define __FIVectorView_1_int_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define __FIVectorView_1_int_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define __FIVectorView_1_int_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 

#define __FIVectorView_1_int_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 

#define __FIVectorView_1_int_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 


#define __FIVectorView_1_int_GetAt(This,index,item)	\
    ( (This)->lpVtbl -> GetAt(This,index,item) ) 

#define __FIVectorView_1_int_get_Size(This,size)	\
    ( (This)->lpVtbl -> get_Size(This,size) ) 

#define __FIVectorView_1_int_IndexOf(This,item,index,found)	\
    ( (This)->lpVtbl -> IndexOf(This,item,index,found) ) 

#define __FIVectorView_1_int_GetMany(This,startIndex,capacity,items,actual)	\
    ( (This)->lpVtbl -> GetMany(This,startIndex,capacity,items,actual) ) 

#endif /* COBJMACROS */



#endif // ____FIVectorView_1_int_INTERFACE_DEFINED__


#if !defined(____FIIterator_1_UINT64_INTERFACE_DEFINED__)
#define ____FIIterator_1_UINT64_INTERFACE_DEFINED__

typedef interface __FIIterator_1_UINT64 __FIIterator_1_UINT64;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIIterator_1_UINT64;

typedef struct __FIIterator_1_UINT64Vtbl
{
    BEGIN_INTERFACE

    HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
        __RPC__in __FIIterator_1_UINT64 * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);
    ULONG ( STDMETHODCALLTYPE *AddRef )(__RPC__in __FIIterator_1_UINT64 * This);
    ULONG ( STDMETHODCALLTYPE *Release )(__RPC__in __FIIterator_1_UINT64 * This);
    HRESULT ( STDMETHODCALLTYPE *GetIids )(__RPC__in __FIIterator_1_UINT64 * This,
        /* [out] */ __RPC__out ULONG *iidCount,
        /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);

    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(__RPC__in __FIIterator_1_UINT64 * This, /* [out] */ __RPC__deref_out_opt HSTRING *className);
    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(__RPC__in __FIIterator_1_UINT64 * This, /* [out] */ __RPC__out TrustLevel *trustLevel);

    /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Current )(__RPC__in __FIIterator_1_UINT64 * This, /* [retval][out] */ __RPC__out unsigned __int64 *current);
    /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_HasCurrent )(__RPC__in __FIIterator_1_UINT64 * This, /* [retval][out] */ __RPC__out boolean *hasCurrent);
    HRESULT ( STDMETHODCALLTYPE *MoveNext )(__RPC__in __FIIterator_1_UINT64 * This, /* [retval][out] */ __RPC__out boolean *hasCurrent);
    HRESULT ( STDMETHODCALLTYPE *GetMany )(__RPC__in __FIIterator_1_UINT64 * This,
        /* [in] */ unsigned int capacity,
        /* [size_is][length_is][out] */ __RPC__out_ecount_part(capacity, *actual) unsigned __int64 *items,
        /* [retval][out] */ __RPC__out unsigned int *actual);

    END_INTERFACE
} __FIIterator_1_UINT64Vtbl;

interface __FIIterator_1_UINT64
{
    CONST_VTBL struct __FIIterator_1_UINT64Vtbl *lpVtbl;
};



#ifdef COBJMACROS


#define __FIIterator_1_UINT64_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define __FIIterator_1_UINT64_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define __FIIterator_1_UINT64_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define __FIIterator_1_UINT64_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 

#define __FIIterator_1_UINT64_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 

#define __FIIterator_1_UINT64_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 


#define __FIIterator_1_UINT64_get_Current(This,current)	\
    ( (This)->lpVtbl -> get_Current(This,current) ) 

#define __FIIterator_1_UINT64_get_HasCurrent(This,hasCurrent)	\
    ( (This)->lpVtbl -> get_HasCurrent(This,hasCurrent) ) 

#define __FIIterator_1_UINT64_MoveNext(This,hasCurrent)	\
    ( (This)->lpVtbl -> MoveNext(This,hasCurrent) ) 

#define __FIIterator_1_UINT64_GetMany(This,capacity,items,actual)	\
    ( (This)->lpVtbl -> GetMany(This,capacity,items,actual) ) 

#endif /* COBJMACROS */


#endif // ____FIIterator_1_UINT64_INTERFACE_DEFINED__


#if !defined(____FIIterable_1_UINT64_INTERFACE_DEFINED__)
#define ____FIIterable_1_UINT64_INTERFACE_DEFINED__

typedef interface __FIIterable_1_UINT64 __FIIterable_1_UINT64;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIIterable_1_UINT64;

typedef  struct __FIIterable_1_UINT64Vtbl
{
    BEGIN_INTERFACE

    HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
        __RPC__in __FIIterable_1_UINT64 * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);

    ULONG ( STDMETHODCALLTYPE *AddRef )(__RPC__in __FIIterable_1_UINT64 * This);

    ULONG ( STDMETHODCALLTYPE *Release )(__RPC__in __FIIterable_1_UINT64 * This);

    HRESULT ( STDMETHODCALLTYPE *GetIids )(__RPC__in __FIIterable_1_UINT64 * This,
                                           /* [out] */ __RPC__out ULONG *iidCount,
                                           /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);

    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(__RPC__in __FIIterable_1_UINT64 * This, /* [out] */ __RPC__deref_out_opt HSTRING *className);

    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(__RPC__in __FIIterable_1_UINT64 * This, /* [out] */ __RPC__out TrustLevel *trustLevel);

    HRESULT ( STDMETHODCALLTYPE *First )(__RPC__in __FIIterable_1_UINT64 * This, /* [retval][out] */ __RPC__deref_out_opt __FIIterator_1_UINT64 **first);

    END_INTERFACE
} __FIIterable_1_UINT64Vtbl;

interface __FIIterable_1_UINT64
{
    CONST_VTBL struct __FIIterable_1_UINT64Vtbl *lpVtbl;
};

#ifdef COBJMACROS

#define __FIIterable_1_UINT64_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define __FIIterable_1_UINT64_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define __FIIterable_1_UINT64_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define __FIIterable_1_UINT64_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 

#define __FIIterable_1_UINT64_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 

#define __FIIterable_1_UINT64_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 


#define __FIIterable_1_UINT64_First(This,first)	\
    ( (This)->lpVtbl -> First(This,first) ) 

#endif /* COBJMACROS */


#endif // ____FIIterable_1_UINT64_INTERFACE_DEFINED__


#if !defined(____FIVectorView_1_UINT64_INTERFACE_DEFINED__)
#define ____FIVectorView_1_UINT64_INTERFACE_DEFINED__

typedef interface __FIVectorView_1_UINT64 __FIVectorView_1_UINT64;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIVectorView_1_UINT64;

typedef struct __FIVectorView_1_UINT64Vtbl
{
    BEGIN_INTERFACE

    HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
        __RPC__in __FIVectorView_1_UINT64 * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);

    ULONG ( STDMETHODCALLTYPE *AddRef )( __RPC__in __FIVectorView_1_UINT64 * This);

    ULONG ( STDMETHODCALLTYPE *Release )( __RPC__in __FIVectorView_1_UINT64 * This);

    HRESULT ( STDMETHODCALLTYPE *GetIids )( __RPC__in __FIVectorView_1_UINT64 * This,
                                            /* [out] */ __RPC__out ULONG *iidCount,
                                            /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);

    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )( 
        __RPC__in __FIVectorView_1_UINT64 * This,
            /* [out] */ __RPC__deref_out_opt HSTRING *className);

    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )( 
        __RPC__in __FIVectorView_1_UINT64 * This,
            /* [out] */ __RPC__out TrustLevel *trustLevel);

    HRESULT ( STDMETHODCALLTYPE *GetAt )( 
                                         __RPC__in __FIVectorView_1_UINT64 * This,
                                         /* [in] */ unsigned int index,
                                         /* [retval][out] */ __RPC__out unsigned __int64 *item);

        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Size )( 
            __RPC__in __FIVectorView_1_UINT64 * This,
            /* [retval][out] */ __RPC__out unsigned int *size);

        HRESULT ( STDMETHODCALLTYPE *IndexOf )( 
                                               __RPC__in __FIVectorView_1_UINT64 * This,
            /* [in] */ unsigned __int64 item,
            /* [out] */ __RPC__out unsigned int *index,
            /* [retval][out] */ __RPC__out boolean *found);

        HRESULT ( STDMETHODCALLTYPE *GetMany )( 
                                               __RPC__in __FIVectorView_1_UINT64 * This,
            /* [in] */ unsigned int startIndex,
            /* [in] */ unsigned int capacity,
            /* [size_is][length_is][out] */ __RPC__out_ecount_part(capacity, *actual) unsigned __int64 *items,
            /* [retval][out] */ __RPC__out unsigned int *actual);

        END_INTERFACE
} __FIVectorView_1_UINT64Vtbl;

interface __FIVectorView_1_UINT64
{
    CONST_VTBL struct __FIVectorView_1_UINT64Vtbl *lpVtbl;
};



#ifdef COBJMACROS


#define __FIVectorView_1_UINT64_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define __FIVectorView_1_UINT64_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define __FIVectorView_1_UINT64_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define __FIVectorView_1_UINT64_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 

#define __FIVectorView_1_UINT64_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 

#define __FIVectorView_1_UINT64_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 


#define __FIVectorView_1_UINT64_GetAt(This,index,item)	\
    ( (This)->lpVtbl -> GetAt(This,index,item) ) 

#define __FIVectorView_1_UINT64_get_Size(This,size)	\
    ( (This)->lpVtbl -> get_Size(This,size) ) 

#define __FIVectorView_1_UINT64_IndexOf(This,item,index,found)	\
    ( (This)->lpVtbl -> IndexOf(This,item,index,found) ) 

#define __FIVectorView_1_UINT64_GetMany(This,startIndex,capacity,items,actual)	\
    ( (This)->lpVtbl -> GetMany(This,startIndex,capacity,items,actual) ) 

#endif /* COBJMACROS */



#endif // ____FIVectorView_1_UINT64_INTERFACE_DEFINED__


#if !defined(____FIIterator_1_boolean_INTERFACE_DEFINED__)
#define ____FIIterator_1_boolean_INTERFACE_DEFINED__

typedef interface __FIIterator_1_boolean __FIIterator_1_boolean;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIIterator_1_boolean;

typedef struct __FIIterator_1_booleanVtbl
{
    BEGIN_INTERFACE

    HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
        __RPC__in __FIIterator_1_boolean * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);
    ULONG ( STDMETHODCALLTYPE *AddRef )(__RPC__in __FIIterator_1_boolean * This);
    ULONG ( STDMETHODCALLTYPE *Release )(__RPC__in __FIIterator_1_boolean * This);
    HRESULT ( STDMETHODCALLTYPE *GetIids )(__RPC__in __FIIterator_1_boolean * This,
        /* [out] */ __RPC__out ULONG *iidCount,
        /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);

    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(__RPC__in __FIIterator_1_boolean * This, /* [out] */ __RPC__deref_out_opt HSTRING *className);
    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(__RPC__in __FIIterator_1_boolean * This, /* [out] */ __RPC__out TrustLevel *trustLevel);

    /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Current )(__RPC__in __FIIterator_1_boolean * This, /* [retval][out] */ __RPC__out boolean *current);
    /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_HasCurrent )(__RPC__in __FIIterator_1_boolean * This, /* [retval][out] */ __RPC__out boolean *hasCurrent);
    HRESULT ( STDMETHODCALLTYPE *MoveNext )(__RPC__in __FIIterator_1_boolean * This, /* [retval][out] */ __RPC__out boolean *hasCurrent);
    HRESULT ( STDMETHODCALLTYPE *GetMany )(__RPC__in __FIIterator_1_boolean * This,
        /* [in] */ unsigned int capacity,
        /* [size_is][length_is][out] */ __RPC__out_ecount_part(capacity, *actual) boolean *items,
        /* [retval][out] */ __RPC__out unsigned int *actual);

    END_INTERFACE
} __FIIterator_1_booleanVtbl;

interface __FIIterator_1_boolean
{
    CONST_VTBL struct __FIIterator_1_booleanVtbl *lpVtbl;
};



#ifdef COBJMACROS


#define __FIIterator_1_boolean_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define __FIIterator_1_boolean_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define __FIIterator_1_boolean_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define __FIIterator_1_boolean_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 

#define __FIIterator_1_boolean_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 

#define __FIIterator_1_boolean_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 


#define __FIIterator_1_boolean_get_Current(This,current)	\
    ( (This)->lpVtbl -> get_Current(This,current) ) 

#define __FIIterator_1_boolean_get_HasCurrent(This,hasCurrent)	\
    ( (This)->lpVtbl -> get_HasCurrent(This,hasCurrent) ) 

#define __FIIterator_1_boolean_MoveNext(This,hasCurrent)	\
    ( (This)->lpVtbl -> MoveNext(This,hasCurrent) ) 

#define __FIIterator_1_boolean_GetMany(This,capacity,items,actual)	\
    ( (This)->lpVtbl -> GetMany(This,capacity,items,actual) ) 

#endif /* COBJMACROS */


#endif // ____FIIterator_1_boolean_INTERFACE_DEFINED__


#if !defined(____FIIterable_1_boolean_INTERFACE_DEFINED__)
#define ____FIIterable_1_boolean_INTERFACE_DEFINED__

typedef interface __FIIterable_1_boolean __FIIterable_1_boolean;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIIterable_1_boolean;

typedef  struct __FIIterable_1_booleanVtbl
{
    BEGIN_INTERFACE

    HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
        __RPC__in __FIIterable_1_boolean * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);

    ULONG ( STDMETHODCALLTYPE *AddRef )(__RPC__in __FIIterable_1_boolean * This);

    ULONG ( STDMETHODCALLTYPE *Release )(__RPC__in __FIIterable_1_boolean * This);

    HRESULT ( STDMETHODCALLTYPE *GetIids )(__RPC__in __FIIterable_1_boolean * This,
                                           /* [out] */ __RPC__out ULONG *iidCount,
                                           /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);

    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(__RPC__in __FIIterable_1_boolean * This, /* [out] */ __RPC__deref_out_opt HSTRING *className);

    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(__RPC__in __FIIterable_1_boolean * This, /* [out] */ __RPC__out TrustLevel *trustLevel);

    HRESULT ( STDMETHODCALLTYPE *First )(__RPC__in __FIIterable_1_boolean * This, /* [retval][out] */ __RPC__deref_out_opt __FIIterator_1_boolean **first);

    END_INTERFACE
} __FIIterable_1_booleanVtbl;

interface __FIIterable_1_boolean
{
    CONST_VTBL struct __FIIterable_1_booleanVtbl *lpVtbl;
};

#ifdef COBJMACROS

#define __FIIterable_1_boolean_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define __FIIterable_1_boolean_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define __FIIterable_1_boolean_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define __FIIterable_1_boolean_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 

#define __FIIterable_1_boolean_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 

#define __FIIterable_1_boolean_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 


#define __FIIterable_1_boolean_First(This,first)	\
    ( (This)->lpVtbl -> First(This,first) ) 

#endif /* COBJMACROS */


#endif // ____FIIterable_1_boolean_INTERFACE_DEFINED__


#if !defined(____FIVectorView_1_boolean_INTERFACE_DEFINED__)
#define ____FIVectorView_1_boolean_INTERFACE_DEFINED__

typedef interface __FIVectorView_1_boolean __FIVectorView_1_boolean;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIVectorView_1_boolean;

typedef struct __FIVectorView_1_booleanVtbl
{
    BEGIN_INTERFACE

    HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
        __RPC__in __FIVectorView_1_boolean * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);

    ULONG ( STDMETHODCALLTYPE *AddRef )( __RPC__in __FIVectorView_1_boolean * This);

    ULONG ( STDMETHODCALLTYPE *Release )( __RPC__in __FIVectorView_1_boolean * This);

    HRESULT ( STDMETHODCALLTYPE *GetIids )( __RPC__in __FIVectorView_1_boolean * This,
                                            /* [out] */ __RPC__out ULONG *iidCount,
                                            /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);

    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )( 
        __RPC__in __FIVectorView_1_boolean * This,
            /* [out] */ __RPC__deref_out_opt HSTRING *className);

    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )( 
        __RPC__in __FIVectorView_1_boolean * This,
            /* [out] */ __RPC__out TrustLevel *trustLevel);

    HRESULT ( STDMETHODCALLTYPE *GetAt )( 
                                         __RPC__in __FIVectorView_1_boolean * This,
                                         /* [in] */ unsigned int index,
                                         /* [retval][out] */ __RPC__out boolean *item);

        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Size )( 
            __RPC__in __FIVectorView_1_boolean * This,
            /* [retval][out] */ __RPC__out unsigned int *size);

        HRESULT ( STDMETHODCALLTYPE *IndexOf )( 
                                               __RPC__in __FIVectorView_1_boolean * This,
            /* [in] */ boolean item,
            /* [out] */ __RPC__out unsigned int *index,
            /* [retval][out] */ __RPC__out boolean *found);

        HRESULT ( STDMETHODCALLTYPE *GetMany )( 
                                               __RPC__in __FIVectorView_1_boolean * This,
            /* [in] */ unsigned int startIndex,
            /* [in] */ unsigned int capacity,
            /* [size_is][length_is][out] */ __RPC__out_ecount_part(capacity, *actual) boolean *items,
            /* [retval][out] */ __RPC__out unsigned int *actual);

        END_INTERFACE
} __FIVectorView_1_booleanVtbl;

interface __FIVectorView_1_boolean
{
    CONST_VTBL struct __FIVectorView_1_booleanVtbl *lpVtbl;
};



#ifdef COBJMACROS


#define __FIVectorView_1_boolean_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define __FIVectorView_1_boolean_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define __FIVectorView_1_boolean_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define __FIVectorView_1_boolean_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 

#define __FIVectorView_1_boolean_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 

#define __FIVectorView_1_boolean_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 


#define __FIVectorView_1_boolean_GetAt(This,index,item)	\
    ( (This)->lpVtbl -> GetAt(This,index,item) ) 

#define __FIVectorView_1_boolean_get_Size(This,size)	\
    ( (This)->lpVtbl -> get_Size(This,size) ) 

#define __FIVectorView_1_boolean_IndexOf(This,item,index,found)	\
    ( (This)->lpVtbl -> IndexOf(This,item,index,found) ) 

#define __FIVectorView_1_boolean_GetMany(This,startIndex,capacity,items,actual)	\
    ( (This)->lpVtbl -> GetMany(This,startIndex,capacity,items,actual) ) 

#endif /* COBJMACROS */



#endif // ____FIVectorView_1_boolean_INTERFACE_DEFINED__


#if !defined(____FIIterator_1_double_INTERFACE_DEFINED__)
#define ____FIIterator_1_double_INTERFACE_DEFINED__

typedef interface __FIIterator_1_double __FIIterator_1_double;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIIterator_1_double;

typedef struct __FIIterator_1_doubleVtbl
{
    BEGIN_INTERFACE

    HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
        __RPC__in __FIIterator_1_double * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);
    ULONG ( STDMETHODCALLTYPE *AddRef )(__RPC__in __FIIterator_1_double * This);
    ULONG ( STDMETHODCALLTYPE *Release )(__RPC__in __FIIterator_1_double * This);
    HRESULT ( STDMETHODCALLTYPE *GetIids )(__RPC__in __FIIterator_1_double * This,
        /* [out] */ __RPC__out ULONG *iidCount,
        /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);

    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(__RPC__in __FIIterator_1_double * This, /* [out] */ __RPC__deref_out_opt HSTRING *className);
    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(__RPC__in __FIIterator_1_double * This, /* [out] */ __RPC__out TrustLevel *trustLevel);

    /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Current )(__RPC__in __FIIterator_1_double * This, /* [retval][out] */ __RPC__out double *current);
    /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_HasCurrent )(__RPC__in __FIIterator_1_double * This, /* [retval][out] */ __RPC__out boolean *hasCurrent);
    HRESULT ( STDMETHODCALLTYPE *MoveNext )(__RPC__in __FIIterator_1_double * This, /* [retval][out] */ __RPC__out boolean *hasCurrent);
    HRESULT ( STDMETHODCALLTYPE *GetMany )(__RPC__in __FIIterator_1_double * This,
        /* [in] */ unsigned int capacity,
        /* [size_is][length_is][out] */ __RPC__out_ecount_part(capacity, *actual) double *items,
        /* [retval][out] */ __RPC__out unsigned int *actual);

    END_INTERFACE
} __FIIterator_1_doubleVtbl;

interface __FIIterator_1_double
{
    CONST_VTBL struct __FIIterator_1_doubleVtbl *lpVtbl;
};



#ifdef COBJMACROS


#define __FIIterator_1_double_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define __FIIterator_1_double_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define __FIIterator_1_double_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define __FIIterator_1_double_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 

#define __FIIterator_1_double_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 

#define __FIIterator_1_double_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 


#define __FIIterator_1_double_get_Current(This,current)	\
    ( (This)->lpVtbl -> get_Current(This,current) ) 

#define __FIIterator_1_double_get_HasCurrent(This,hasCurrent)	\
    ( (This)->lpVtbl -> get_HasCurrent(This,hasCurrent) ) 

#define __FIIterator_1_double_MoveNext(This,hasCurrent)	\
    ( (This)->lpVtbl -> MoveNext(This,hasCurrent) ) 

#define __FIIterator_1_double_GetMany(This,capacity,items,actual)	\
    ( (This)->lpVtbl -> GetMany(This,capacity,items,actual) ) 

#endif /* COBJMACROS */


#endif // ____FIIterator_1_double_INTERFACE_DEFINED__


#if !defined(____FIIterable_1_double_INTERFACE_DEFINED__)
#define ____FIIterable_1_double_INTERFACE_DEFINED__

typedef interface __FIIterable_1_double __FIIterable_1_double;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIIterable_1_double;

typedef  struct __FIIterable_1_doubleVtbl
{
    BEGIN_INTERFACE

    HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
        __RPC__in __FIIterable_1_double * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);

    ULONG ( STDMETHODCALLTYPE *AddRef )(__RPC__in __FIIterable_1_double * This);

    ULONG ( STDMETHODCALLTYPE *Release )(__RPC__in __FIIterable_1_double * This);

    HRESULT ( STDMETHODCALLTYPE *GetIids )(__RPC__in __FIIterable_1_double * This,
                                           /* [out] */ __RPC__out ULONG *iidCount,
                                           /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);

    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(__RPC__in __FIIterable_1_double * This, /* [out] */ __RPC__deref_out_opt HSTRING *className);

    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(__RPC__in __FIIterable_1_double * This, /* [out] */ __RPC__out TrustLevel *trustLevel);

    HRESULT ( STDMETHODCALLTYPE *First )(__RPC__in __FIIterable_1_double * This, /* [retval][out] */ __RPC__deref_out_opt __FIIterator_1_double **first);

    END_INTERFACE
} __FIIterable_1_doubleVtbl;

interface __FIIterable_1_double
{
    CONST_VTBL struct __FIIterable_1_doubleVtbl *lpVtbl;
};

#ifdef COBJMACROS

#define __FIIterable_1_double_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define __FIIterable_1_double_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define __FIIterable_1_double_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define __FIIterable_1_double_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 

#define __FIIterable_1_double_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 

#define __FIIterable_1_double_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 


#define __FIIterable_1_double_First(This,first)	\
    ( (This)->lpVtbl -> First(This,first) ) 

#endif /* COBJMACROS */


#endif // ____FIIterable_1_double_INTERFACE_DEFINED__


#if !defined(____FIVectorView_1_double_INTERFACE_DEFINED__)
#define ____FIVectorView_1_double_INTERFACE_DEFINED__

typedef interface __FIVectorView_1_double __FIVectorView_1_double;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIVectorView_1_double;

typedef struct __FIVectorView_1_doubleVtbl
{
    BEGIN_INTERFACE

    HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
        __RPC__in __FIVectorView_1_double * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);

    ULONG ( STDMETHODCALLTYPE *AddRef )( __RPC__in __FIVectorView_1_double * This);

    ULONG ( STDMETHODCALLTYPE *Release )( __RPC__in __FIVectorView_1_double * This);

    HRESULT ( STDMETHODCALLTYPE *GetIids )( __RPC__in __FIVectorView_1_double * This,
                                            /* [out] */ __RPC__out ULONG *iidCount,
                                            /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);

    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )( 
        __RPC__in __FIVectorView_1_double * This,
            /* [out] */ __RPC__deref_out_opt HSTRING *className);

    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )( 
        __RPC__in __FIVectorView_1_double * This,
            /* [out] */ __RPC__out TrustLevel *trustLevel);

    HRESULT ( STDMETHODCALLTYPE *GetAt )( 
                                         __RPC__in __FIVectorView_1_double * This,
                                         /* [in] */ unsigned int index,
                                         /* [retval][out] */ __RPC__out double *item);

        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Size )( 
            __RPC__in __FIVectorView_1_double * This,
            /* [retval][out] */ __RPC__out unsigned int *size);

        HRESULT ( STDMETHODCALLTYPE *IndexOf )( 
                                               __RPC__in __FIVectorView_1_double * This,
            /* [in] */ double item,
            /* [out] */ __RPC__out unsigned int *index,
            /* [retval][out] */ __RPC__out boolean *found);

        HRESULT ( STDMETHODCALLTYPE *GetMany )( 
                                               __RPC__in __FIVectorView_1_double * This,
            /* [in] */ unsigned int startIndex,
            /* [in] */ unsigned int capacity,
            /* [size_is][length_is][out] */ __RPC__out_ecount_part(capacity, *actual) double *items,
            /* [retval][out] */ __RPC__out unsigned int *actual);

        END_INTERFACE
} __FIVectorView_1_doubleVtbl;

interface __FIVectorView_1_double
{
    CONST_VTBL struct __FIVectorView_1_doubleVtbl *lpVtbl;
};



#ifdef COBJMACROS


#define __FIVectorView_1_double_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define __FIVectorView_1_double_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define __FIVectorView_1_double_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define __FIVectorView_1_double_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 

#define __FIVectorView_1_double_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 

#define __FIVectorView_1_double_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 


#define __FIVectorView_1_double_GetAt(This,index,item)	\
    ( (This)->lpVtbl -> GetAt(This,index,item) ) 

#define __FIVectorView_1_double_get_Size(This,size)	\
    ( (This)->lpVtbl -> get_Size(This,size) ) 

#define __FIVectorView_1_double_IndexOf(This,item,index,found)	\
    ( (This)->lpVtbl -> IndexOf(This,item,index,found) ) 

#define __FIVectorView_1_double_GetMany(This,startIndex,capacity,items,actual)	\
    ( (This)->lpVtbl -> GetMany(This,startIndex,capacity,items,actual) ) 

#endif /* COBJMACROS */



#endif // ____FIVectorView_1_double_INTERFACE_DEFINED__


#if !defined(____FIIterator_1_HSTRING_INTERFACE_DEFINED__)
#define ____FIIterator_1_HSTRING_INTERFACE_DEFINED__

typedef interface __FIIterator_1_HSTRING __FIIterator_1_HSTRING;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIIterator_1_HSTRING;

typedef struct __FIIterator_1_HSTRINGVtbl
{
    BEGIN_INTERFACE

    HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
        __RPC__in __FIIterator_1_HSTRING * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);
    ULONG ( STDMETHODCALLTYPE *AddRef )(__RPC__in __FIIterator_1_HSTRING * This);
    ULONG ( STDMETHODCALLTYPE *Release )(__RPC__in __FIIterator_1_HSTRING * This);
    HRESULT ( STDMETHODCALLTYPE *GetIids )(__RPC__in __FIIterator_1_HSTRING * This,
        /* [out] */ __RPC__out ULONG *iidCount,
        /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);

    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(__RPC__in __FIIterator_1_HSTRING * This, /* [out] */ __RPC__deref_out_opt HSTRING *className);
    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(__RPC__in __FIIterator_1_HSTRING * This, /* [out] */ __RPC__out TrustLevel *trustLevel);

    /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Current )(__RPC__in __FIIterator_1_HSTRING * This, /* [retval][out] */ __RPC__out HSTRING *current);
    /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_HasCurrent )(__RPC__in __FIIterator_1_HSTRING * This, /* [retval][out] */ __RPC__out boolean *hasCurrent);
    HRESULT ( STDMETHODCALLTYPE *MoveNext )(__RPC__in __FIIterator_1_HSTRING * This, /* [retval][out] */ __RPC__out boolean *hasCurrent);
    HRESULT ( STDMETHODCALLTYPE *GetMany )(__RPC__in __FIIterator_1_HSTRING * This,
        /* [in] */ unsigned int capacity,
        /* [size_is][length_is][out] */ __RPC__out_ecount_part(capacity, *actual) HSTRING *items,
        /* [retval][out] */ __RPC__out unsigned int *actual);

    END_INTERFACE
} __FIIterator_1_HSTRINGVtbl;

interface __FIIterator_1_HSTRING
{
    CONST_VTBL struct __FIIterator_1_HSTRINGVtbl *lpVtbl;
};



#ifdef COBJMACROS


#define __FIIterator_1_HSTRING_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define __FIIterator_1_HSTRING_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define __FIIterator_1_HSTRING_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define __FIIterator_1_HSTRING_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 

#define __FIIterator_1_HSTRING_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 

#define __FIIterator_1_HSTRING_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 


#define __FIIterator_1_HSTRING_get_Current(This,current)	\
    ( (This)->lpVtbl -> get_Current(This,current) ) 

#define __FIIterator_1_HSTRING_get_HasCurrent(This,hasCurrent)	\
    ( (This)->lpVtbl -> get_HasCurrent(This,hasCurrent) ) 

#define __FIIterator_1_HSTRING_MoveNext(This,hasCurrent)	\
    ( (This)->lpVtbl -> MoveNext(This,hasCurrent) ) 

#define __FIIterator_1_HSTRING_GetMany(This,capacity,items,actual)	\
    ( (This)->lpVtbl -> GetMany(This,capacity,items,actual) ) 

#endif /* COBJMACROS */


#endif // ____FIIterator_1_HSTRING_INTERFACE_DEFINED__


#if !defined(____FIIterable_1_HSTRING_INTERFACE_DEFINED__)
#define ____FIIterable_1_HSTRING_INTERFACE_DEFINED__

typedef interface __FIIterable_1_HSTRING __FIIterable_1_HSTRING;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIIterable_1_HSTRING;

typedef  struct __FIIterable_1_HSTRINGVtbl
{
    BEGIN_INTERFACE

    HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
        __RPC__in __FIIterable_1_HSTRING * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);

    ULONG ( STDMETHODCALLTYPE *AddRef )(__RPC__in __FIIterable_1_HSTRING * This);

    ULONG ( STDMETHODCALLTYPE *Release )(__RPC__in __FIIterable_1_HSTRING * This);

    HRESULT ( STDMETHODCALLTYPE *GetIids )(__RPC__in __FIIterable_1_HSTRING * This,
                                           /* [out] */ __RPC__out ULONG *iidCount,
                                           /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);

    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(__RPC__in __FIIterable_1_HSTRING * This, /* [out] */ __RPC__deref_out_opt HSTRING *className);

    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(__RPC__in __FIIterable_1_HSTRING * This, /* [out] */ __RPC__out TrustLevel *trustLevel);

    HRESULT ( STDMETHODCALLTYPE *First )(__RPC__in __FIIterable_1_HSTRING * This, /* [retval][out] */ __RPC__deref_out_opt __FIIterator_1_HSTRING **first);

    END_INTERFACE
} __FIIterable_1_HSTRINGVtbl;

interface __FIIterable_1_HSTRING
{
    CONST_VTBL struct __FIIterable_1_HSTRINGVtbl *lpVtbl;
};

#ifdef COBJMACROS

#define __FIIterable_1_HSTRING_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define __FIIterable_1_HSTRING_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define __FIIterable_1_HSTRING_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define __FIIterable_1_HSTRING_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 

#define __FIIterable_1_HSTRING_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 

#define __FIIterable_1_HSTRING_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 


#define __FIIterable_1_HSTRING_First(This,first)	\
    ( (This)->lpVtbl -> First(This,first) ) 

#endif /* COBJMACROS */


#endif // ____FIIterable_1_HSTRING_INTERFACE_DEFINED__


#if !defined(____FIVectorView_1_HSTRING_INTERFACE_DEFINED__)
#define ____FIVectorView_1_HSTRING_INTERFACE_DEFINED__

typedef interface __FIVectorView_1_HSTRING __FIVectorView_1_HSTRING;

//  Declare the parameterized interface IID.
EXTERN_C const IID IID___FIVectorView_1_HSTRING;

typedef struct __FIVectorView_1_HSTRINGVtbl
{
    BEGIN_INTERFACE

    HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
        __RPC__in __FIVectorView_1_HSTRING * This,
        /* [in] */ __RPC__in REFIID riid,
        /* [annotation][iid_is][out] */ 
        _COM_Outptr_  void **ppvObject);

    ULONG ( STDMETHODCALLTYPE *AddRef )( __RPC__in __FIVectorView_1_HSTRING * This);

    ULONG ( STDMETHODCALLTYPE *Release )( __RPC__in __FIVectorView_1_HSTRING * This);

    HRESULT ( STDMETHODCALLTYPE *GetIids )( __RPC__in __FIVectorView_1_HSTRING * This,
                                            /* [out] */ __RPC__out ULONG *iidCount,
                                            /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids);

    HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )( 
        __RPC__in __FIVectorView_1_HSTRING * This,
            /* [out] */ __RPC__deref_out_opt HSTRING *className);

    HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )( 
        __RPC__in __FIVectorView_1_HSTRING * This,
            /* [out] */ __RPC__out TrustLevel *trustLevel);

    HRESULT ( STDMETHODCALLTYPE *GetAt )( 
                                         __RPC__in __FIVectorView_1_HSTRING * This,
                                         /* [in] */ unsigned int index,
                                         /* [retval][out] */ __RPC__out HSTRING *item);

        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Size )( 
            __RPC__in __FIVectorView_1_HSTRING * This,
            /* [retval][out] */ __RPC__out unsigned int *size);

        HRESULT ( STDMETHODCALLTYPE *IndexOf )( 
                                               __RPC__in __FIVectorView_1_HSTRING * This,
            /* [in] */ HSTRING item,
            /* [out] */ __RPC__out unsigned int *index,
            /* [retval][out] */ __RPC__out boolean *found);

        HRESULT ( STDMETHODCALLTYPE *GetMany )( 
                                               __RPC__in __FIVectorView_1_HSTRING * This,
            /* [in] */ unsigned int startIndex,
            /* [in] */ unsigned int capacity,
            /* [size_is][length_is][out] */ __RPC__out_ecount_part(capacity, *actual) HSTRING *items,
            /* [retval][out] */ __RPC__out unsigned int *actual);

        END_INTERFACE
} __FIVectorView_1_HSTRINGVtbl;

interface __FIVectorView_1_HSTRING
{
    CONST_VTBL struct __FIVectorView_1_HSTRINGVtbl *lpVtbl;
};



#ifdef COBJMACROS


#define __FIVectorView_1_HSTRING_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define __FIVectorView_1_HSTRING_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define __FIVectorView_1_HSTRING_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define __FIVectorView_1_HSTRING_GetIids(This,iidCount,iids)	\
    ( (This)->lpVtbl -> GetIids(This,iidCount,iids) ) 

#define __FIVectorView_1_HSTRING_GetRuntimeClassName(This,className)	\
    ( (This)->lpVtbl -> GetRuntimeClassName(This,className) ) 

#define __FIVectorView_1_HSTRING_GetTrustLevel(This,trustLevel)	\
    ( (This)->lpVtbl -> GetTrustLevel(This,trustLevel) ) 


#define __FIVectorView_1_HSTRING_GetAt(This,index,item)	\
    ( (This)->lpVtbl -> GetAt(This,index,item) ) 

#define __FIVectorView_1_HSTRING_get_Size(This,size)	\
    ( (This)->lpVtbl -> get_Size(This,size) ) 

#define __FIVectorView_1_HSTRING_IndexOf(This,item,index,found)	\
    ( (This)->lpVtbl -> IndexOf(This,item,index,found) ) 

#define __FIVectorView_1_HSTRING_GetMany(This,startIndex,capacity,items,actual)	\
    ( (This)->lpVtbl -> GetMany(This,startIndex,capacity,items,actual) ) 

#endif /* COBJMACROS */



#endif // ____FIVectorView_1_HSTRING_INTERFACE_DEFINED__




#pragma warning (push)
#pragma warning (disable:4668) 
#pragma warning (disable:4001) 
#pragma once 
#pragma warning (pop)



/*
 *
 * Struct Microsoft.AI.MachineLearning.LearningModelFeatureKind
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 */

#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
/* [v1_enum, contract] */
enum __x_ABI_CMicrosoft_CAI_CMachineLearning_CLearningModelFeatureKind
{
    LearningModelFeatureKind_Tensor = 0,
    LearningModelFeatureKind_Sequence,
    LearningModelFeatureKind_Map,
    LearningModelFeatureKind_Image,
};
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ILearningModelFeatureDescriptor
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ILearningModelFeatureDescriptor:HRESULT get_Name(String*);HRESULT get_Description(String*);HRESULT get_Kind(Microsoft.AI.MachineLearning.LearningModelFeatureKind*);HRESULT get_IsRequired(Boolean*);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ILearningModelFeatureDescriptor[] = L"Microsoft.AI.MachineLearning.ILearningModelFeatureDescriptor";
/* [uuid("d21da0b6-b8b9-534b-97da-b3f0e0ffbac2"), object, contract] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptorVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
/* [propget] */HRESULT ( STDMETHODCALLTYPE *get_Name )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor * This,
        /* [out, retval] */HSTRING * value
        );
    /* [propget] */HRESULT ( STDMETHODCALLTYPE *get_Description )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor * This,
        /* [out, retval] */HSTRING * value
        );
    /* [propget] */HRESULT ( STDMETHODCALLTYPE *get_Kind )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor * This,
        /* [out, retval] */enum __x_ABI_CMicrosoft_CAI_CMachineLearning_CLearningModelFeatureKind * value
        );
    /* [propget] */HRESULT ( STDMETHODCALLTYPE *get_IsRequired )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor * This,
        /* [out, retval] */boolean * value
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptorVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptorVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor_get_Name(This,value) \
    ( (This)->lpVtbl->get_Name(This,value) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor_get_Description(This,value) \
    ( (This)->lpVtbl->get_Description(This,value) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor_get_Kind(This,value) \
    ( (This)->lpVtbl->get_Kind(This,value) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor_get_IsRequired(This,value) \
    ( (This)->lpVtbl->get_IsRequired(This,value) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ILearningModelOperatorProvider
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelOperatorProvider_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelOperatorProvider_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ILearningModelOperatorProvider[] = L"Microsoft.AI.MachineLearning.ILearningModelOperatorProvider";
/* [object, contract, uuid("ae066239-6b19-4509-be3e-502ba40203b3")] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelOperatorProviderVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelOperatorProvider * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelOperatorProvider * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelOperatorProvider * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelOperatorProvider * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelOperatorProvider * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelOperatorProvider * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelOperatorProviderVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelOperatorProvider
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelOperatorProviderVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelOperatorProvider_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelOperatorProvider_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelOperatorProvider_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelOperatorProvider_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelOperatorProvider_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelOperatorProvider_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelOperatorProvider;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelOperatorProvider_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000



/*
 *
 * Interface Microsoft.AI.MachineLearning.ILearningModel
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.LearningModel
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ILearningModel:HRESULT get_Author(String*);HRESULT get_Name(String*);HRESULT get_Domain(String*);HRESULT get_Description(String*);HRESULT get_Version(Int64*);HRESULT get_Metadata(Windows.Foundation.Collections.IMapView`2<String, String>**);HRESULT get_InputFeatures(Windows.Foundation.Collections.IVectorView`1<Microsoft.AI.MachineLearning.ILearningModelFeatureDescriptor>**);HRESULT get_OutputFeatures(Windows.Foundation.Collections.IVectorView`1<Microsoft.AI.MachineLearning.ILearningModelFeatureDescriptor>**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ILearningModel[] = L"Microsoft.AI.MachineLearning.ILearningModel";
/* [uuid("3e81e9c7-6af9-5bc0-941a-0b1857abb2d8"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
/* [propget] */HRESULT ( STDMETHODCALLTYPE *get_Author )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel * This,
        /* [out, retval] */HSTRING * value
        );
    /* [propget] */HRESULT ( STDMETHODCALLTYPE *get_Name )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel * This,
        /* [out, retval] */HSTRING * value
        );
    /* [propget] */HRESULT ( STDMETHODCALLTYPE *get_Domain )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel * This,
        /* [out, retval] */HSTRING * value
        );
    /* [propget] */HRESULT ( STDMETHODCALLTYPE *get_Description )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel * This,
        /* [out, retval] */HSTRING * value
        );
    /* [propget] */HRESULT ( STDMETHODCALLTYPE *get_Version )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel * This,
        /* [out, retval] */__int64 * value
        );
    /* [propget] */HRESULT ( STDMETHODCALLTYPE *get_Metadata )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel * This,
        /* [out, retval] */__FIMapView_2_HSTRING_HSTRING * * value
        );
    /* [propget] */HRESULT ( STDMETHODCALLTYPE *get_InputFeatures )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel * This,
        /* [out, retval] */__FIVectorView_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor * * value
        );
    /* [propget] */HRESULT ( STDMETHODCALLTYPE *get_OutputFeatures )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel * This,
        /* [out, retval] */__FIVectorView_1_Microsoft__CAI__CMachineLearning__CILearningModelFeatureDescriptor * * value
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel_get_Author(This,value) \
    ( (This)->lpVtbl->get_Author(This,value) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel_get_Name(This,value) \
    ( (This)->lpVtbl->get_Name(This,value) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel_get_Domain(This,value) \
    ( (This)->lpVtbl->get_Domain(This,value) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel_get_Description(This,value) \
    ( (This)->lpVtbl->get_Description(This,value) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel_get_Version(This,value) \
    ( (This)->lpVtbl->get_Version(This,value) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel_get_Metadata(This,value) \
    ( (This)->lpVtbl->get_Metadata(This,value) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel_get_InputFeatures(This,value) \
    ( (This)->lpVtbl->get_InputFeatures(This,value) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel_get_OutputFeatures(This,value) \
    ( (This)->lpVtbl->get_OutputFeatures(This,value) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ILearningModelStatics
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.LearningModel
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ILearningModelStatics:HRESULT LoadFromStorageFileAsync(Windows.Storage.IStorageFile*,Windows.Foundation.IAsyncOperation`1<Microsoft.AI.MachineLearning.LearningModel>**);HRESULT LoadFromStreamAsync(Windows.Storage.Streams.IRandomAccessStreamReference*,Windows.Foundation.IAsyncOperation`1<Microsoft.AI.MachineLearning.LearningModel>**);HRESULT LoadFromFilePath(String,Microsoft.AI.MachineLearning.LearningModel**);HRESULT LoadFromStream(Windows.Storage.Streams.IRandomAccessStreamReference*,Microsoft.AI.MachineLearning.LearningModel**);HRESULT LoadFromStorageFileWithOperatorProviderAsync(Windows.Storage.IStorageFile*,Microsoft.AI.MachineLearning.ILearningModelOperatorProvider*,Windows.Foundation.IAsyncOperation`1<Microsoft.AI.MachineLearning.LearningModel>**);HRESULT LoadFromStreamWithOperatorProviderAsync(Windows.Storage.Streams.IRandomAccessStreamReference*,Microsoft.AI.MachineLearning.ILearningModelOperatorProvider*,Windows.Foundation.IAsyncOperation`1<Microsoft.AI.MachineLearning.LearningModel>**);HRESULT LoadFromFilePathWithOperatorProvider(String,Microsoft.AI.MachineLearning.ILearningModelOperatorProvider*,Microsoft.AI.MachineLearning.LearningModel**);HRESULT LoadFromStreamWithOperatorProvider(Windows.Storage.Streams.IRandomAccessStreamReference*,Microsoft.AI.MachineLearning.ILearningModelOperatorProvider*,Microsoft.AI.MachineLearning.LearningModel**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ILearningModelStatics[] = L"Microsoft.AI.MachineLearning.ILearningModelStatics";
/* [uuid("ae329fdc-7a71-5b31-bbd9-bf0af125a6c3"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStaticsVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
/* [overload, remote_async] */HRESULT ( STDMETHODCALLTYPE *LoadFromStorageFileAsync )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics * This,
        /* [in] */__x_ABI_CWindows_CStorage_CIStorageFile * modelFile,
        /* [out, retval] */__FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModel * * operation
        );
    /* [overload, remote_async] */HRESULT ( STDMETHODCALLTYPE *LoadFromStreamAsync )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics * This,
        /* [in] */__x_ABI_CWindows_CStorage_CStreams_CIRandomAccessStreamReference * modelStream,
        /* [out, retval] */__FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModel * * operation
        );
    /* [overload] */HRESULT ( STDMETHODCALLTYPE *LoadFromFilePath )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics * This,
        /* [in] */HSTRING filePath,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel * * result
        );
    /* [overload] */HRESULT ( STDMETHODCALLTYPE *LoadFromStream )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics * This,
        /* [in] */__x_ABI_CWindows_CStorage_CStreams_CIRandomAccessStreamReference * modelStream,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel * * result
        );
    /* [overload, remote_async] */HRESULT ( STDMETHODCALLTYPE *LoadFromStorageFileWithOperatorProviderAsync )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics * This,
        /* [in] */__x_ABI_CWindows_CStorage_CIStorageFile * modelFile,
        /* [in] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelOperatorProvider * operatorProvider,
        /* [out, retval] */__FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModel * * operation
        );
    /* [overload, remote_async] */HRESULT ( STDMETHODCALLTYPE *LoadFromStreamWithOperatorProviderAsync )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics * This,
        /* [in] */__x_ABI_CWindows_CStorage_CStreams_CIRandomAccessStreamReference * modelStream,
        /* [in] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelOperatorProvider * operatorProvider,
        /* [out, retval] */__FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModel * * operation
        );
    /* [overload] */HRESULT ( STDMETHODCALLTYPE *LoadFromFilePathWithOperatorProvider )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics * This,
        /* [in] */HSTRING filePath,
        /* [in] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelOperatorProvider * operatorProvider,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel * * result
        );
    /* [overload] */HRESULT ( STDMETHODCALLTYPE *LoadFromStreamWithOperatorProvider )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics * This,
        /* [in] */__x_ABI_CWindows_CStorage_CStreams_CIRandomAccessStreamReference * modelStream,
        /* [in] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelOperatorProvider * operatorProvider,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel * * result
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStaticsVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStaticsVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics_LoadFromStorageFileAsync(This,modelFile,operation) \
    ( (This)->lpVtbl->LoadFromStorageFileAsync(This,modelFile,operation) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics_LoadFromStreamAsync(This,modelStream,operation) \
    ( (This)->lpVtbl->LoadFromStreamAsync(This,modelStream,operation) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics_LoadFromFilePath(This,filePath,result) \
    ( (This)->lpVtbl->LoadFromFilePath(This,filePath,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics_LoadFromStream(This,modelStream,result) \
    ( (This)->lpVtbl->LoadFromStream(This,modelStream,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics_LoadFromStorageFileWithOperatorProviderAsync(This,modelFile,operatorProvider,operation) \
    ( (This)->lpVtbl->LoadFromStorageFileWithOperatorProviderAsync(This,modelFile,operatorProvider,operation) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics_LoadFromStreamWithOperatorProviderAsync(This,modelStream,operatorProvider,operation) \
    ( (This)->lpVtbl->LoadFromStreamWithOperatorProviderAsync(This,modelStream,operatorProvider,operation) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics_LoadFromFilePathWithOperatorProvider(This,filePath,operatorProvider,result) \
    ( (This)->lpVtbl->LoadFromFilePathWithOperatorProvider(This,filePath,operatorProvider,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics_LoadFromStreamWithOperatorProvider(This,modelStream,operatorProvider,result) \
    ( (This)->lpVtbl->LoadFromStreamWithOperatorProvider(This,modelStream,operatorProvider,result) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelStatics_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Class Microsoft.AI.MachineLearning.LearningModel
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * RuntimeClass contains static methods.
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ILearningModelStatics interface starting with version 1.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.ILearningModel ** Default Interface **
 *    Windows.Foundation.IClosable
 *
 * Class Threading Model:  Both Single and Multi Threaded Apartment
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_LearningModel_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_LearningModel_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_LearningModel[] = L"Microsoft.AI.MachineLearning.LearningModel";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Struct Microsoft.AI.MachineLearning.LearningModelDeviceKind
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 */

#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
/* [v1_enum, contract] */
enum __x_ABI_CMicrosoft_CAI_CMachineLearning_CLearningModelDeviceKind
{
    LearningModelDeviceKind_Default = 0,
    LearningModelDeviceKind_Cpu,
    LearningModelDeviceKind_DirectX,
    LearningModelDeviceKind_DirectXHighPerformance,
    LearningModelDeviceKind_DirectXMinPower,
};
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000



/*
 *
 * Interface Microsoft.AI.MachineLearning.ILearningModelDevice
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.LearningModelDevice
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ILearningModelDevice:HRESULT get_AdapterId(Windows.Graphics.DisplayAdapterId*);HRESULT get_Direct3D11Device(Windows.Graphics.DirectX.Direct3D11.IDirect3DDevice**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDevice_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDevice_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ILearningModelDevice[] = L"Microsoft.AI.MachineLearning.ILearningModelDevice";
/* [uuid("b56e79aa-89ee-5e25-8c5c-e32b68b51447"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDevice * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDevice * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDevice * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDevice * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDevice * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDevice * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
/* [propget] */HRESULT ( STDMETHODCALLTYPE *get_AdapterId )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDevice * This,
        /* [out, retval] */struct __x_ABI_CWindows_CGraphics_CDisplayAdapterId * value
        );
    /* [propget] */HRESULT ( STDMETHODCALLTYPE *get_Direct3D11Device )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDevice * This,
        /* [out, retval] */__x_ABI_CWindows_CGraphics_CDirectX_CDirect3D11_CIDirect3DDevice * * value
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDevice
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDevice_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDevice_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDevice_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDevice_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDevice_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDevice_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDevice_get_AdapterId(This,value) \
    ( (This)->lpVtbl->get_AdapterId(This,value) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDevice_get_Direct3D11Device(This,value) \
    ( (This)->lpVtbl->get_Direct3D11Device(This,value) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDevice;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDevice_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ILearningModelDeviceStatics
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.LearningModelDevice
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ILearningModelDeviceStatics:HRESULT CreateFromDirect3D11Device(Windows.Graphics.DirectX.Direct3D11.IDirect3DDevice*,Microsoft.AI.MachineLearning.LearningModelDevice**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceStatics_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceStatics_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ILearningModelDeviceStatics[] = L"Microsoft.AI.MachineLearning.ILearningModelDeviceStatics";
/* [uuid("a622a3cd-0e4e-5f65-9a6f-06ab1b325582"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceStaticsVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceStatics * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceStatics * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceStatics * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceStatics * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceStatics * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceStatics * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
HRESULT ( STDMETHODCALLTYPE *CreateFromDirect3D11Device )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceStatics * This,
        /* [in] */__x_ABI_CWindows_CGraphics_CDirectX_CDirect3D11_CIDirect3DDevice * device,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDevice * * result
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceStaticsVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceStatics
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceStaticsVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceStatics_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceStatics_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceStatics_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceStatics_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceStatics_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceStatics_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceStatics_CreateFromDirect3D11Device(This,device,result) \
    ( (This)->lpVtbl->CreateFromDirect3D11Device(This,device,result) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceStatics;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceStatics_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ILearningModelDeviceFactory
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.LearningModelDevice
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ILearningModelDeviceFactory:HRESULT Create(Microsoft.AI.MachineLearning.LearningModelDeviceKind,Microsoft.AI.MachineLearning.LearningModelDevice**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceFactory_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceFactory_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ILearningModelDeviceFactory[] = L"Microsoft.AI.MachineLearning.ILearningModelDeviceFactory";
/* [uuid("0ad28f1e-8cdf-5dbc-8af3-51090e34e780"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceFactoryVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceFactory * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceFactory * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceFactory * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceFactory * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceFactory * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceFactory * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
HRESULT ( STDMETHODCALLTYPE *Create )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceFactory * This,
        /* [in] */enum __x_ABI_CMicrosoft_CAI_CMachineLearning_CLearningModelDeviceKind deviceKind,
        /* [retval, out] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDevice * * value
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceFactoryVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceFactory
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceFactoryVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceFactory_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceFactory_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceFactory_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceFactory_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceFactory_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceFactory_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceFactory_Create(This,deviceKind,value) \
    ( (This)->lpVtbl->Create(This,deviceKind,value) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceFactory;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDeviceFactory_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Class Microsoft.AI.MachineLearning.LearningModelDevice
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * RuntimeClass can be activated.
 *   Type can be activated via the Microsoft.AI.MachineLearning.ILearningModelDeviceFactory interface starting with version 1.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *
 * RuntimeClass contains static methods.
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ILearningModelDeviceStatics interface starting with version 1.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.ILearningModelDevice ** Default Interface **
 *
 * Class Threading Model:  Both Single and Multi Threaded Apartment
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_LearningModelDevice_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_LearningModelDevice_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_LearningModelDevice[] = L"Microsoft.AI.MachineLearning.LearningModelDevice";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000



/*
 *
 * Interface Microsoft.AI.MachineLearning.ILearningModelEvaluationResult
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.LearningModelEvaluationResult
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ILearningModelEvaluationResult:HRESULT get_CorrelationId(String*);HRESULT get_ErrorStatus(Int32*);HRESULT get_Succeeded(Boolean*);HRESULT get_Outputs(Windows.Foundation.Collections.IMapView`2<String, Object>**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelEvaluationResult_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelEvaluationResult_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ILearningModelEvaluationResult[] = L"Microsoft.AI.MachineLearning.ILearningModelEvaluationResult";
/* [uuid("576800d1-be6c-50ab-bd7e-a7e01508f3bc"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelEvaluationResultVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelEvaluationResult * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelEvaluationResult * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelEvaluationResult * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelEvaluationResult * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelEvaluationResult * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelEvaluationResult * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
/* [propget] */HRESULT ( STDMETHODCALLTYPE *get_CorrelationId )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelEvaluationResult * This,
        /* [out, retval] */HSTRING * value
        );
    /* [propget] */HRESULT ( STDMETHODCALLTYPE *get_ErrorStatus )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelEvaluationResult * This,
        /* [out, retval] */int * value
        );
    /* [propget] */HRESULT ( STDMETHODCALLTYPE *get_Succeeded )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelEvaluationResult * This,
        /* [out, retval] */boolean * value
        );
    /* [propget] */HRESULT ( STDMETHODCALLTYPE *get_Outputs )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelEvaluationResult * This,
        /* [out, retval] */__FIMapView_2_HSTRING_IInspectable * * value
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelEvaluationResultVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelEvaluationResult
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelEvaluationResultVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelEvaluationResult_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelEvaluationResult_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelEvaluationResult_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelEvaluationResult_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelEvaluationResult_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelEvaluationResult_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelEvaluationResult_get_CorrelationId(This,value) \
    ( (This)->lpVtbl->get_CorrelationId(This,value) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelEvaluationResult_get_ErrorStatus(This,value) \
    ( (This)->lpVtbl->get_ErrorStatus(This,value) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelEvaluationResult_get_Succeeded(This,value) \
    ( (This)->lpVtbl->get_Succeeded(This,value) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelEvaluationResult_get_Outputs(This,value) \
    ( (This)->lpVtbl->get_Outputs(This,value) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelEvaluationResult;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelEvaluationResult_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Class Microsoft.AI.MachineLearning.LearningModelEvaluationResult
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.ILearningModelEvaluationResult ** Default Interface **
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_LearningModelEvaluationResult_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_LearningModelEvaluationResult_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_LearningModelEvaluationResult[] = L"Microsoft.AI.MachineLearning.LearningModelEvaluationResult";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000



/*
 *
 * Interface Microsoft.AI.MachineLearning.ILearningModelSessionOptions
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 2.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.LearningModelSessionOptions
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ILearningModelSessionOptions:HRESULT get_BatchSizeOverride(UInt32*);HRESULT put_BatchSizeOverride(UInt32);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ILearningModelSessionOptions[] = L"Microsoft.AI.MachineLearning.ILearningModelSessionOptions";
/* [uuid("88968c30-4c69-5135-869a-26a0f4f9007f"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptionsVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
/* [propget] */HRESULT ( STDMETHODCALLTYPE *get_BatchSizeOverride )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions * This,
        /* [out, retval] */unsigned int * value
        );
    /* [propput] */HRESULT ( STDMETHODCALLTYPE *put_BatchSizeOverride )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions * This,
        /* [in] */unsigned int value
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptionsVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptionsVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions_get_BatchSizeOverride(This,value) \
    ( (This)->lpVtbl->get_BatchSizeOverride(This,value) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions_put_BatchSizeOverride(This,value) \
    ( (This)->lpVtbl->put_BatchSizeOverride(This,value) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ILearningModelSessionOptions2
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 3.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.LearningModelSessionOptions
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ILearningModelSessionOptions2:HRESULT get_CloseModelOnSessionCreation(Boolean*);HRESULT put_CloseModelOnSessionCreation(Boolean);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x30000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions2_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions2_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ILearningModelSessionOptions2[] = L"Microsoft.AI.MachineLearning.ILearningModelSessionOptions2";
/* [uuid("46d6729a-e3a1-5412-a45b-d26460eda2db"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions2Vtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions2 * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions2 * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions2 * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions2 * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions2 * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions2 * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
/* [propget] */HRESULT ( STDMETHODCALLTYPE *get_CloseModelOnSessionCreation )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions2 * This,
        /* [out, retval] */boolean * value
        );
    /* [propput] */HRESULT ( STDMETHODCALLTYPE *put_CloseModelOnSessionCreation )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions2 * This,
        /* [in] */boolean value
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions2Vtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions2
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions2Vtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions2_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions2_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions2_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions2_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions2_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions2_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions2_get_CloseModelOnSessionCreation(This,value) \
    ( (This)->lpVtbl->get_CloseModelOnSessionCreation(This,value) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions2_put_CloseModelOnSessionCreation(This,value) \
    ( (This)->lpVtbl->put_CloseModelOnSessionCreation(This,value) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions2;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions2_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x30000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ILearningModelSessionOptions3
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 4.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.LearningModelSessionOptions
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ILearningModelSessionOptions3:HRESULT OverrideNamedDimension(String,UInt32);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x40000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions3_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions3_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ILearningModelSessionOptions3[] = L"Microsoft.AI.MachineLearning.ILearningModelSessionOptions3";
/* [uuid("d346a34c-6639-5118-b38a-e3360ef1f69d"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions3Vtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions3 * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions3 * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions3 * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions3 * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions3 * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions3 * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
HRESULT ( STDMETHODCALLTYPE *OverrideNamedDimension )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions3 * This,
        /* [in] */HSTRING name,
        /* [in] */unsigned int dimension
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions3Vtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions3
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions3Vtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions3_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions3_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions3_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions3_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions3_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions3_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions3_OverrideNamedDimension(This,name,dimension) \
    ( (This)->lpVtbl->OverrideNamedDimension(This,name,dimension) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions3;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions3_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x40000


/*
 *
 * Class Microsoft.AI.MachineLearning.LearningModelSessionOptions
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 2.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * RuntimeClass can be activated.
 *   Type can be activated via RoActivateInstance starting with version 2.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.ILearningModelSessionOptions ** Default Interface **
 *    Microsoft.AI.MachineLearning.ILearningModelSessionOptions2
 *    Microsoft.AI.MachineLearning.ILearningModelSessionOptions3
 *
 * Class Threading Model:  Both Single and Multi Threaded Apartment
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_LearningModelSessionOptions_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_LearningModelSessionOptions_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_LearningModelSessionOptions[] = L"Microsoft.AI.MachineLearning.LearningModelSessionOptions";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000



/*
 *
 * Interface Microsoft.AI.MachineLearning.ILearningModelSession
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.LearningModelSession
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ILearningModelSession:HRESULT get_Model(Microsoft.AI.MachineLearning.LearningModel**);HRESULT get_Device(Microsoft.AI.MachineLearning.LearningModelDevice**);HRESULT get_EvaluationProperties(Windows.Foundation.Collections.IPropertySet**);HRESULT EvaluateAsync(Microsoft.AI.MachineLearning.LearningModelBinding*,String,Windows.Foundation.IAsyncOperation`1<Microsoft.AI.MachineLearning.LearningModelEvaluationResult>**);HRESULT EvaluateFeaturesAsync(Windows.Foundation.Collections.IMap`2<String, Object>*,String,Windows.Foundation.IAsyncOperation`1<Microsoft.AI.MachineLearning.LearningModelEvaluationResult>**);HRESULT Evaluate(Microsoft.AI.MachineLearning.LearningModelBinding*,String,Microsoft.AI.MachineLearning.LearningModelEvaluationResult**);HRESULT EvaluateFeatures(Windows.Foundation.Collections.IMap`2<String, Object>*,String,Microsoft.AI.MachineLearning.LearningModelEvaluationResult**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ILearningModelSession[] = L"Microsoft.AI.MachineLearning.ILearningModelSession";
/* [uuid("326031fc-c12d-5c0e-8209-4ad32fa41047"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
/* [propget] */HRESULT ( STDMETHODCALLTYPE *get_Model )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession * This,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel * * value
        );
    /* [propget] */HRESULT ( STDMETHODCALLTYPE *get_Device )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession * This,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDevice * * value
        );
    /* [propget] */HRESULT ( STDMETHODCALLTYPE *get_EvaluationProperties )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession * This,
        /* [out, retval] */__x_ABI_CWindows_CFoundation_CCollections_CIPropertySet * * value
        );
    /* [remote_async] */HRESULT ( STDMETHODCALLTYPE *EvaluateAsync )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession * This,
        /* [in] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBinding * bindings,
        /* [in] */HSTRING correlationId,
        /* [out, retval] */__FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult * * operation
        );
    /* [remote_async] */HRESULT ( STDMETHODCALLTYPE *EvaluateFeaturesAsync )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession * This,
        /* [in] */__FIMap_2_HSTRING_IInspectable * features,
        /* [in] */HSTRING correlationId,
        /* [out, retval] */__FIAsyncOperation_1_Microsoft__CAI__CMachineLearning__CLearningModelEvaluationResult * * operation
        );
    HRESULT ( STDMETHODCALLTYPE *Evaluate )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession * This,
        /* [in] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBinding * bindings,
        /* [in] */HSTRING correlationId,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelEvaluationResult * * result
        );
    HRESULT ( STDMETHODCALLTYPE *EvaluateFeatures )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession * This,
        /* [in] */__FIMap_2_HSTRING_IInspectable * features,
        /* [in] */HSTRING correlationId,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelEvaluationResult * * result
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession_get_Model(This,value) \
    ( (This)->lpVtbl->get_Model(This,value) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession_get_Device(This,value) \
    ( (This)->lpVtbl->get_Device(This,value) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession_get_EvaluationProperties(This,value) \
    ( (This)->lpVtbl->get_EvaluationProperties(This,value) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession_EvaluateAsync(This,bindings,correlationId,operation) \
    ( (This)->lpVtbl->EvaluateAsync(This,bindings,correlationId,operation) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession_EvaluateFeaturesAsync(This,features,correlationId,operation) \
    ( (This)->lpVtbl->EvaluateFeaturesAsync(This,features,correlationId,operation) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession_Evaluate(This,bindings,correlationId,result) \
    ( (This)->lpVtbl->Evaluate(This,bindings,correlationId,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession_EvaluateFeatures(This,features,correlationId,result) \
    ( (This)->lpVtbl->EvaluateFeatures(This,features,correlationId,result) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ILearningModelSessionFactory
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.LearningModelSession
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ILearningModelSessionFactory:HRESULT CreateFromModel(Microsoft.AI.MachineLearning.LearningModel*,Microsoft.AI.MachineLearning.LearningModelSession**);HRESULT CreateFromModelOnDevice(Microsoft.AI.MachineLearning.LearningModel*,Microsoft.AI.MachineLearning.LearningModelDevice*,Microsoft.AI.MachineLearning.LearningModelSession**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ILearningModelSessionFactory[] = L"Microsoft.AI.MachineLearning.ILearningModelSessionFactory";
/* [uuid("d7d86c54-d03d-5ae3-a958-fe952b640620"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactoryVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
HRESULT ( STDMETHODCALLTYPE *CreateFromModel )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory * This,
        /* [in] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel * model,
        /* [retval, out] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession * * value
        );
    HRESULT ( STDMETHODCALLTYPE *CreateFromModelOnDevice )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory * This,
        /* [in] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel * model,
        /* [in] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDevice * deviceToRunOn,
        /* [retval, out] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession * * value
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactoryVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactoryVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory_CreateFromModel(This,model,value) \
    ( (This)->lpVtbl->CreateFromModel(This,model,value) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory_CreateFromModelOnDevice(This,model,deviceToRunOn,value) \
    ( (This)->lpVtbl->CreateFromModelOnDevice(This,model,deviceToRunOn,value) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ILearningModelSessionFactory2
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 2.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.LearningModelSession
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ILearningModelSessionFactory2:HRESULT CreateFromModelOnDeviceWithSessionOptions(Microsoft.AI.MachineLearning.LearningModel*,Microsoft.AI.MachineLearning.LearningModelDevice*,Microsoft.AI.MachineLearning.LearningModelSessionOptions*,Microsoft.AI.MachineLearning.LearningModelSession**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory2_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory2_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ILearningModelSessionFactory2[] = L"Microsoft.AI.MachineLearning.ILearningModelSessionFactory2";
/* [uuid("1a25cd87-032c-5775-bdaa-852915a025a9"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory2Vtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory2 * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory2 * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory2 * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory2 * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory2 * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory2 * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
HRESULT ( STDMETHODCALLTYPE *CreateFromModelOnDeviceWithSessionOptions )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory2 * This,
        /* [in] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModel * model,
        /* [in] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelDevice * deviceToRunOn,
        /* [in] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionOptions * learningModelSessionOptions,
        /* [retval, out] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession * * value
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory2Vtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory2
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory2Vtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory2_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory2_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory2_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory2_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory2_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory2_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory2_CreateFromModelOnDeviceWithSessionOptions(This,model,deviceToRunOn,learningModelSessionOptions,value) \
    ( (This)->lpVtbl->CreateFromModelOnDeviceWithSessionOptions(This,model,deviceToRunOn,learningModelSessionOptions,value) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory2;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSessionFactory2_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000


/*
 *
 * Class Microsoft.AI.MachineLearning.LearningModelSession
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * RuntimeClass can be activated.
 *   Type can be activated via the Microsoft.AI.MachineLearning.ILearningModelSessionFactory2 interface starting with version 2.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *   Type can be activated via the Microsoft.AI.MachineLearning.ILearningModelSessionFactory interface starting with version 1.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.ILearningModelSession ** Default Interface **
 *    Windows.Foundation.IClosable
 *
 * Class Threading Model:  Both Single and Multi Threaded Apartment
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_LearningModelSession_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_LearningModelSession_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_LearningModelSession[] = L"Microsoft.AI.MachineLearning.LearningModelSession";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ILearningModelFeatureValue
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ILearningModelFeatureValue:HRESULT get_Kind(Microsoft.AI.MachineLearning.LearningModelFeatureKind*);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureValue_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureValue_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ILearningModelFeatureValue[] = L"Microsoft.AI.MachineLearning.ILearningModelFeatureValue";
/* [uuid("ae7816ef-cb63-5aa4-81fc-e902a0a6c977"), object, contract] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureValueVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureValue * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureValue * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureValue * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureValue * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureValue * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureValue * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
/* [propget] */HRESULT ( STDMETHODCALLTYPE *get_Kind )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureValue * This,
        /* [out, retval] */enum __x_ABI_CMicrosoft_CAI_CMachineLearning_CLearningModelFeatureKind * value
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureValueVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureValue
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureValueVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureValue_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureValue_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureValue_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureValue_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureValue_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureValue_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureValue_get_Kind(This,value) \
    ( (This)->lpVtbl->get_Kind(This,value) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureValue;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureValue_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ILearningModelBinding
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.LearningModelBinding
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ILearningModelBinding:HRESULT Bind(String,Object*);HRESULT BindWithProperties(String,Object*,Windows.Foundation.Collections.IPropertySet*);HRESULT Clear();
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBinding_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBinding_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ILearningModelBinding[] = L"Microsoft.AI.MachineLearning.ILearningModelBinding";
/* [uuid("2f77b18a-68ae-510f-afb9-5af57446c343"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBindingVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBinding * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBinding * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBinding * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBinding * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBinding * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBinding * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
/* [overload] */HRESULT ( STDMETHODCALLTYPE *Bind )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBinding * This,
        /* [in] */HSTRING name,
        /* [in] */IInspectable * value
        );
    /* [overload] */HRESULT ( STDMETHODCALLTYPE *BindWithProperties )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBinding * This,
        /* [in] */HSTRING name,
        /* [in] */IInspectable * value,
        /* [in] */__x_ABI_CWindows_CFoundation_CCollections_CIPropertySet * props
        );
    HRESULT ( STDMETHODCALLTYPE *Clear )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBinding * This
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBindingVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBinding
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBindingVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBinding_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBinding_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBinding_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBinding_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBinding_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBinding_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBinding_Bind(This,name,value) \
    ( (This)->lpVtbl->Bind(This,name,value) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBinding_BindWithProperties(This,name,value,props) \
    ( (This)->lpVtbl->BindWithProperties(This,name,value,props) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBinding_Clear(This) \
    ( (This)->lpVtbl->Clear(This) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBinding;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBinding_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ILearningModelBindingFactory
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.LearningModelBinding
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ILearningModelBindingFactory:HRESULT CreateFromSession(Microsoft.AI.MachineLearning.LearningModelSession*,Microsoft.AI.MachineLearning.LearningModelBinding**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBindingFactory_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBindingFactory_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ILearningModelBindingFactory[] = L"Microsoft.AI.MachineLearning.ILearningModelBindingFactory";
/* [uuid("ae2f1c97-2fd5-55b9-a05f-53b9dbb4f9e2"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBindingFactoryVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBindingFactory * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBindingFactory * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBindingFactory * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBindingFactory * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBindingFactory * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBindingFactory * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
HRESULT ( STDMETHODCALLTYPE *CreateFromSession )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBindingFactory * This,
        /* [in] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelSession * session,
        /* [retval, out] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBinding * * value
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBindingFactoryVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBindingFactory
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBindingFactoryVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBindingFactory_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBindingFactory_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBindingFactory_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBindingFactory_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBindingFactory_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBindingFactory_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBindingFactory_CreateFromSession(This,session,value) \
    ( (This)->lpVtbl->CreateFromSession(This,session,value) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBindingFactory;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelBindingFactory_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Class Microsoft.AI.MachineLearning.LearningModelBinding
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * RuntimeClass can be activated.
 *   Type can be activated via the Microsoft.AI.MachineLearning.ILearningModelBindingFactory interface starting with version 1.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.ILearningModelBinding ** Default Interface **
 *    Windows.Foundation.Collections.IMapView_2_HSTRING,IInspectable
 *
 * Class Threading Model:  Both Single and Multi Threaded Apartment
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_LearningModelBinding_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_LearningModelBinding_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_LearningModelBinding[] = L"Microsoft.AI.MachineLearning.LearningModelBinding";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Struct Microsoft.AI.MachineLearning.TensorKind
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 */

#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
/* [v1_enum, contract] */
enum __x_ABI_CMicrosoft_CAI_CMachineLearning_CTensorKind
{
    TensorKind_Undefined = 0,
    TensorKind_Float,
    TensorKind_UInt8,
    TensorKind_Int8,
    TensorKind_UInt16,
    TensorKind_Int16,
    TensorKind_Int32,
    TensorKind_Int64,
    TensorKind_String,
    TensorKind_Boolean,
    TensorKind_Float16,
    TensorKind_Double,
    TensorKind_UInt32,
    TensorKind_UInt64,
    TensorKind_Complex64,
    TensorKind_Complex128,
};
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000



/*
 *
 * Interface Microsoft.AI.MachineLearning.IMapFeatureDescriptor
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.MapFeatureDescriptor
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.IMapFeatureDescriptor:HRESULT get_KeyKind(Microsoft.AI.MachineLearning.TensorKind*);HRESULT get_ValueDescriptor(Microsoft.AI.MachineLearning.ILearningModelFeatureDescriptor**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CIMapFeatureDescriptor_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CIMapFeatureDescriptor_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_IMapFeatureDescriptor[] = L"Microsoft.AI.MachineLearning.IMapFeatureDescriptor";
/* [uuid("308161e6-8c55-5db3-8ea7-b70edae7f362"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CIMapFeatureDescriptorVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CIMapFeatureDescriptor * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CIMapFeatureDescriptor * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CIMapFeatureDescriptor * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CIMapFeatureDescriptor * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CIMapFeatureDescriptor * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CIMapFeatureDescriptor * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
/* [propget] */HRESULT ( STDMETHODCALLTYPE *get_KeyKind )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CIMapFeatureDescriptor * This,
        /* [out, retval] */enum __x_ABI_CMicrosoft_CAI_CMachineLearning_CTensorKind * value
        );
    /* [propget] */HRESULT ( STDMETHODCALLTYPE *get_ValueDescriptor )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CIMapFeatureDescriptor * This,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor * * value
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CIMapFeatureDescriptorVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CIMapFeatureDescriptor
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CIMapFeatureDescriptorVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CIMapFeatureDescriptor_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CIMapFeatureDescriptor_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CIMapFeatureDescriptor_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CIMapFeatureDescriptor_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CIMapFeatureDescriptor_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CIMapFeatureDescriptor_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CIMapFeatureDescriptor_get_KeyKind(This,value) \
    ( (This)->lpVtbl->get_KeyKind(This,value) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CIMapFeatureDescriptor_get_ValueDescriptor(This,value) \
    ( (This)->lpVtbl->get_ValueDescriptor(This,value) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CIMapFeatureDescriptor;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CIMapFeatureDescriptor_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Class Microsoft.AI.MachineLearning.MapFeatureDescriptor
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.IMapFeatureDescriptor ** Default Interface **
 *    Microsoft.AI.MachineLearning.ILearningModelFeatureDescriptor
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_MapFeatureDescriptor_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_MapFeatureDescriptor_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_MapFeatureDescriptor[] = L"Microsoft.AI.MachineLearning.MapFeatureDescriptor";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000



/*
 *
 * Interface Microsoft.AI.MachineLearning.ISequenceFeatureDescriptor
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.SequenceFeatureDescriptor
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ISequenceFeatureDescriptor:HRESULT get_ElementDescriptor(Microsoft.AI.MachineLearning.ILearningModelFeatureDescriptor**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CISequenceFeatureDescriptor_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CISequenceFeatureDescriptor_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ISequenceFeatureDescriptor[] = L"Microsoft.AI.MachineLearning.ISequenceFeatureDescriptor";
/* [uuid("bcc21461-ba27-54f0-a5c6-2bb81aca668d"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CISequenceFeatureDescriptorVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CISequenceFeatureDescriptor * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CISequenceFeatureDescriptor * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CISequenceFeatureDescriptor * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CISequenceFeatureDescriptor * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CISequenceFeatureDescriptor * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CISequenceFeatureDescriptor * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
/* [propget] */HRESULT ( STDMETHODCALLTYPE *get_ElementDescriptor )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CISequenceFeatureDescriptor * This,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CILearningModelFeatureDescriptor * * value
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CISequenceFeatureDescriptorVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CISequenceFeatureDescriptor
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CISequenceFeatureDescriptorVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CISequenceFeatureDescriptor_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CISequenceFeatureDescriptor_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CISequenceFeatureDescriptor_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CISequenceFeatureDescriptor_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CISequenceFeatureDescriptor_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CISequenceFeatureDescriptor_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CISequenceFeatureDescriptor_get_ElementDescriptor(This,value) \
    ( (This)->lpVtbl->get_ElementDescriptor(This,value) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CISequenceFeatureDescriptor;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CISequenceFeatureDescriptor_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Class Microsoft.AI.MachineLearning.SequenceFeatureDescriptor
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.ISequenceFeatureDescriptor ** Default Interface **
 *    Microsoft.AI.MachineLearning.ILearningModelFeatureDescriptor
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_SequenceFeatureDescriptor_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_SequenceFeatureDescriptor_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_SequenceFeatureDescriptor[] = L"Microsoft.AI.MachineLearning.SequenceFeatureDescriptor";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000



/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorFeatureDescriptor
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorFeatureDescriptor
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorFeatureDescriptor:HRESULT get_TensorKind(Microsoft.AI.MachineLearning.TensorKind*);HRESULT get_Shape(Windows.Foundation.Collections.IVectorView`1<Int64>**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFeatureDescriptor_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFeatureDescriptor_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorFeatureDescriptor[] = L"Microsoft.AI.MachineLearning.ITensorFeatureDescriptor";
/* [uuid("4de2edbe-6519-51ef-b88b-8d9820449e1b"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFeatureDescriptorVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFeatureDescriptor * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFeatureDescriptor * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFeatureDescriptor * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFeatureDescriptor * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFeatureDescriptor * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFeatureDescriptor * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
/* [propget] */HRESULT ( STDMETHODCALLTYPE *get_TensorKind )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFeatureDescriptor * This,
        /* [out, retval] */enum __x_ABI_CMicrosoft_CAI_CMachineLearning_CTensorKind * value
        );
    /* [propget] */HRESULT ( STDMETHODCALLTYPE *get_Shape )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFeatureDescriptor * This,
        /* [out, retval] */__FIVectorView_1___z__zint64 * * value
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFeatureDescriptorVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFeatureDescriptor
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFeatureDescriptorVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFeatureDescriptor_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFeatureDescriptor_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFeatureDescriptor_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFeatureDescriptor_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFeatureDescriptor_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFeatureDescriptor_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFeatureDescriptor_get_TensorKind(This,value) \
    ( (This)->lpVtbl->get_TensorKind(This,value) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFeatureDescriptor_get_Shape(This,value) \
    ( (This)->lpVtbl->get_Shape(This,value) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFeatureDescriptor;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFeatureDescriptor_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Class Microsoft.AI.MachineLearning.TensorFeatureDescriptor
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.ITensorFeatureDescriptor ** Default Interface **
 *    Microsoft.AI.MachineLearning.ILearningModelFeatureDescriptor
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorFeatureDescriptor_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorFeatureDescriptor_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_TensorFeatureDescriptor[] = L"Microsoft.AI.MachineLearning.TensorFeatureDescriptor";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Struct Microsoft.AI.MachineLearning.LearningModelPixelRange
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 5.0
 *
 *
 */

#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x50000
/* [v1_enum, contract] */
enum __x_ABI_CMicrosoft_CAI_CMachineLearning_CLearningModelPixelRange
{
    LearningModelPixelRange_ZeroTo255 = 0,
    LearningModelPixelRange_ZeroToOne,
    LearningModelPixelRange_MinusOneToOne,
};
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x50000



/*
 *
 * Interface Microsoft.AI.MachineLearning.IImageFeatureDescriptor
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.ImageFeatureDescriptor
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.IImageFeatureDescriptor:HRESULT get_BitmapPixelFormat(Windows.Graphics.Imaging.BitmapPixelFormat*);HRESULT get_BitmapAlphaMode(Windows.Graphics.Imaging.BitmapAlphaMode*);HRESULT get_Width(UInt32*);HRESULT get_Height(UInt32*);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_IImageFeatureDescriptor[] = L"Microsoft.AI.MachineLearning.IImageFeatureDescriptor";
/* [uuid("5c9c1823-77b7-5b43-aafc-9c540c50ff60"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptorVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
/* [propget] */HRESULT ( STDMETHODCALLTYPE *get_BitmapPixelFormat )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor * This,
        /* [out, retval] */enum __x_ABI_CWindows_CGraphics_CImaging_CBitmapPixelFormat * value
        );
    /* [propget] */HRESULT ( STDMETHODCALLTYPE *get_BitmapAlphaMode )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor * This,
        /* [out, retval] */enum __x_ABI_CWindows_CGraphics_CImaging_CBitmapAlphaMode * value
        );
    /* [propget] */HRESULT ( STDMETHODCALLTYPE *get_Width )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor * This,
        /* [out, retval] */unsigned int * value
        );
    /* [propget] */HRESULT ( STDMETHODCALLTYPE *get_Height )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor * This,
        /* [out, retval] */unsigned int * value
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptorVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptorVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor_get_BitmapPixelFormat(This,value) \
    ( (This)->lpVtbl->get_BitmapPixelFormat(This,value) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor_get_BitmapAlphaMode(This,value) \
    ( (This)->lpVtbl->get_BitmapAlphaMode(This,value) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor_get_Width(This,value) \
    ( (This)->lpVtbl->get_Width(This,value) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor_get_Height(This,value) \
    ( (This)->lpVtbl->get_Height(This,value) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.IImageFeatureDescriptor2
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 5.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.ImageFeatureDescriptor
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.IImageFeatureDescriptor2:HRESULT get_PixelRange(Microsoft.AI.MachineLearning.LearningModelPixelRange*);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x50000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor2_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor2_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_IImageFeatureDescriptor2[] = L"Microsoft.AI.MachineLearning.IImageFeatureDescriptor2";
/* [uuid("01f87bfc-90b9-5af7-8cbd-aa9e0e96f4d2"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor2Vtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor2 * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor2 * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor2 * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor2 * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor2 * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor2 * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
/* [propget] */HRESULT ( STDMETHODCALLTYPE *get_PixelRange )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor2 * This,
        /* [out, retval] */enum __x_ABI_CMicrosoft_CAI_CMachineLearning_CLearningModelPixelRange * value
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor2Vtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor2
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor2Vtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor2_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor2_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor2_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor2_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor2_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor2_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor2_get_PixelRange(This,value) \
    ( (This)->lpVtbl->get_PixelRange(This,value) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor2;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureDescriptor2_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x50000


/*
 *
 * Class Microsoft.AI.MachineLearning.ImageFeatureDescriptor
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.IImageFeatureDescriptor ** Default Interface **
 *    Microsoft.AI.MachineLearning.IImageFeatureDescriptor2
 *    Microsoft.AI.MachineLearning.ILearningModelFeatureDescriptor
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_ImageFeatureDescriptor_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_ImageFeatureDescriptor_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_ImageFeatureDescriptor[] = L"Microsoft.AI.MachineLearning.ImageFeatureDescriptor";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensor
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Any object which implements this interface must also implement the following interfaces:
 *     Microsoft.AI.MachineLearning.ILearningModelFeatureValue
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensor:HRESULT get_TensorKind(Microsoft.AI.MachineLearning.TensorKind*);HRESULT get_Shape(Windows.Foundation.Collections.IVectorView`1<Int64>**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensor_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensor_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensor[] = L"Microsoft.AI.MachineLearning.ITensor";
/* [uuid("376f95b3-eb6c-551f-b76b-369e925b4149"), object, contract] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensor * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensor * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensor * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensor * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensor * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensor * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
/* [propget] */HRESULT ( STDMETHODCALLTYPE *get_TensorKind )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensor * This,
        /* [out, retval] */enum __x_ABI_CMicrosoft_CAI_CMachineLearning_CTensorKind * value
        );
    /* [propget] */HRESULT ( STDMETHODCALLTYPE *get_Shape )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensor * This,
        /* [out, retval] */__FIVectorView_1___z__zint64 * * value
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensor
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensor_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensor_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensor_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensor_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensor_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensor_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensor_get_TensorKind(This,value) \
    ( (This)->lpVtbl->get_TensorKind(This,value) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensor_get_Shape(This,value) \
    ( (This)->lpVtbl->get_Shape(This,value) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensor;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensor_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000



/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorFloat
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorFloat
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorFloat:HRESULT GetAsVectorView(Windows.Foundation.Collections.IVectorView`1<Single>**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorFloat[] = L"Microsoft.AI.MachineLearning.ITensorFloat";
/* [uuid("b137fe61-183a-51b1-87de-f5662d9e5f34"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
HRESULT ( STDMETHODCALLTYPE *GetAsVectorView )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat * This,
        /* [out, retval] */__FIVectorView_1_float * * result
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat_GetAsVectorView(This,result) \
    ( (This)->lpVtbl->GetAsVectorView(This,result) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorFloatStatics
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorFloat
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorFloatStatics:HRESULT Create(Microsoft.AI.MachineLearning.TensorFloat**);HRESULT Create2(Windows.Foundation.Collections.IIterable`1<Int64>*,Microsoft.AI.MachineLearning.TensorFloat**);HRESULT CreateFromArray(Windows.Foundation.Collections.IIterable`1<Int64>*,UInt32,Single*,Microsoft.AI.MachineLearning.TensorFloat**);HRESULT CreateFromIterable(Windows.Foundation.Collections.IIterable`1<Int64>*,Windows.Foundation.Collections.IIterable`1<Single>*,Microsoft.AI.MachineLearning.TensorFloat**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorFloatStatics[] = L"Microsoft.AI.MachineLearning.ITensorFloatStatics";
/* [uuid("650fee2e-c9f8-54bd-a895-4d37e2e146d0"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStaticsVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
/* [overload] */HRESULT ( STDMETHODCALLTYPE *Create )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics * This,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat * * result
        );
    /* [overload] */HRESULT ( STDMETHODCALLTYPE *Create2 )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics * This,
        /* [in] */__FIIterable_1___z__zint64 * shape,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat * * result
        );
    HRESULT ( STDMETHODCALLTYPE *CreateFromArray )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics * This,
        /* [in] */__FIIterable_1___z__zint64 * shape,
        /* [in] */unsigned int dataLength,
        /* [size_is(dataLength), in] */float * data,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat * * result
        );
    HRESULT ( STDMETHODCALLTYPE *CreateFromIterable )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics * This,
        /* [in] */__FIIterable_1___z__zint64 * shape,
        /* [in] */__FIIterable_1_float * data,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat * * result
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStaticsVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStaticsVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics_Create(This,result) \
    ( (This)->lpVtbl->Create(This,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics_Create2(This,shape,result) \
    ( (This)->lpVtbl->Create2(This,shape,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics_CreateFromArray(This,shape,dataLength,data,result) \
    ( (This)->lpVtbl->CreateFromArray(This,shape,dataLength,data,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics_CreateFromIterable(This,shape,data,result) \
    ( (This)->lpVtbl->CreateFromIterable(This,shape,data,result) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorFloatStatics2
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 2.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorFloat
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorFloatStatics2:HRESULT CreateFromShapeArrayAndDataArray(UInt32,Int64*,UInt32,Single*,Microsoft.AI.MachineLearning.TensorFloat**);HRESULT CreateFromBuffer(UInt32,Int64*,Windows.Storage.Streams.IBuffer*,Microsoft.AI.MachineLearning.TensorFloat**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics2_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics2_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorFloatStatics2[] = L"Microsoft.AI.MachineLearning.ITensorFloatStatics2";
/* [uuid("73e9f04f-5f05-515a-8179-8ffe46a639cf"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics2Vtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics2 * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics2 * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics2 * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics2 * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics2 * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics2 * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
HRESULT ( STDMETHODCALLTYPE *CreateFromShapeArrayAndDataArray )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics2 * This,
        /* [in] */unsigned int shapeLength,
        /* [size_is(shapeLength), in] */__int64 * shape,
        /* [in] */unsigned int dataLength,
        /* [size_is(dataLength), in] */float * data,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat * * result
        );
    HRESULT ( STDMETHODCALLTYPE *CreateFromBuffer )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics2 * This,
        /* [in] */unsigned int shapeLength,
        /* [size_is(shapeLength), in] */__int64 * shape,
        /* [in] */__x_ABI_CWindows_CStorage_CStreams_CIBuffer * buffer,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat * * result
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics2Vtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics2
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics2Vtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics2_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics2_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics2_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics2_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics2_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics2_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics2_CreateFromShapeArrayAndDataArray(This,shapeLength,shape,dataLength,data,result) \
    ( (This)->lpVtbl->CreateFromShapeArrayAndDataArray(This,shapeLength,shape,dataLength,data,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics2_CreateFromBuffer(This,shapeLength,shape,buffer,result) \
    ( (This)->lpVtbl->CreateFromBuffer(This,shapeLength,shape,buffer,result) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics2;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloatStatics2_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000


/*
 *
 * Class Microsoft.AI.MachineLearning.TensorFloat
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * RuntimeClass contains static methods.
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorFloatStatics2 interface starting with version 2.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorFloatStatics interface starting with version 1.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.ITensorFloat ** Default Interface **
 *    Microsoft.AI.MachineLearning.ITensor
 *    Windows.Foundation.IMemoryBuffer
 *
 * Class Threading Model:  Both Single and Multi Threaded Apartment
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorFloat_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorFloat_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_TensorFloat[] = L"Microsoft.AI.MachineLearning.TensorFloat";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000



/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorFloat16Bit
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorFloat16Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorFloat16Bit:HRESULT GetAsVectorView(Windows.Foundation.Collections.IVectorView`1<Single>**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16Bit_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16Bit_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorFloat16Bit[] = L"Microsoft.AI.MachineLearning.ITensorFloat16Bit";
/* [uuid("5b653d83-e84f-59bc-a489-014ad5c332a2"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16Bit * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16Bit * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16Bit * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16Bit * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16Bit * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16Bit * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
HRESULT ( STDMETHODCALLTYPE *GetAsVectorView )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16Bit * This,
        /* [out, retval] */__FIVectorView_1_float * * result
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16Bit
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16Bit_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16Bit_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16Bit_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16Bit_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16Bit_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16Bit_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16Bit_GetAsVectorView(This,result) \
    ( (This)->lpVtbl->GetAsVectorView(This,result) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16Bit;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16Bit_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorFloat16BitStatics
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorFloat16Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorFloat16BitStatics:HRESULT Create(Microsoft.AI.MachineLearning.TensorFloat16Bit**);HRESULT Create2(Windows.Foundation.Collections.IIterable`1<Int64>*,Microsoft.AI.MachineLearning.TensorFloat16Bit**);HRESULT CreateFromArray(Windows.Foundation.Collections.IIterable`1<Int64>*,UInt32,Single*,Microsoft.AI.MachineLearning.TensorFloat16Bit**);HRESULT CreateFromIterable(Windows.Foundation.Collections.IIterable`1<Int64>*,Windows.Foundation.Collections.IIterable`1<Single>*,Microsoft.AI.MachineLearning.TensorFloat16Bit**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorFloat16BitStatics[] = L"Microsoft.AI.MachineLearning.ITensorFloat16BitStatics";
/* [uuid("87dfb838-eb96-5d5e-bb03-efab0b6a7b0d"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStaticsVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
/* [overload] */HRESULT ( STDMETHODCALLTYPE *Create )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics * This,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16Bit * * result
        );
    /* [overload] */HRESULT ( STDMETHODCALLTYPE *Create2 )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics * This,
        /* [in] */__FIIterable_1___z__zint64 * shape,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16Bit * * result
        );
    HRESULT ( STDMETHODCALLTYPE *CreateFromArray )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics * This,
        /* [in] */__FIIterable_1___z__zint64 * shape,
        /* [in] */unsigned int dataLength,
        /* [size_is(dataLength), in] */float * data,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16Bit * * result
        );
    HRESULT ( STDMETHODCALLTYPE *CreateFromIterable )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics * This,
        /* [in] */__FIIterable_1___z__zint64 * shape,
        /* [in] */__FIIterable_1_float * data,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16Bit * * result
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStaticsVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStaticsVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics_Create(This,result) \
    ( (This)->lpVtbl->Create(This,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics_Create2(This,shape,result) \
    ( (This)->lpVtbl->Create2(This,shape,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics_CreateFromArray(This,shape,dataLength,data,result) \
    ( (This)->lpVtbl->CreateFromArray(This,shape,dataLength,data,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics_CreateFromIterable(This,shape,data,result) \
    ( (This)->lpVtbl->CreateFromIterable(This,shape,data,result) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorFloat16BitStatics2
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 2.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorFloat16Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorFloat16BitStatics2:HRESULT CreateFromShapeArrayAndDataArray(UInt32,Int64*,UInt32,Single*,Microsoft.AI.MachineLearning.TensorFloat16Bit**);HRESULT CreateFromBuffer(UInt32,Int64*,Windows.Storage.Streams.IBuffer*,Microsoft.AI.MachineLearning.TensorFloat16Bit**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics2_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics2_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorFloat16BitStatics2[] = L"Microsoft.AI.MachineLearning.ITensorFloat16BitStatics2";
/* [uuid("4af7175c-b626-50a3-a169-555de125bc2d"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics2Vtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics2 * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics2 * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics2 * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics2 * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics2 * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics2 * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
HRESULT ( STDMETHODCALLTYPE *CreateFromShapeArrayAndDataArray )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics2 * This,
        /* [in] */unsigned int shapeLength,
        /* [size_is(shapeLength), in] */__int64 * shape,
        /* [in] */unsigned int dataLength,
        /* [size_is(dataLength), in] */float * data,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16Bit * * result
        );
    HRESULT ( STDMETHODCALLTYPE *CreateFromBuffer )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics2 * This,
        /* [in] */unsigned int shapeLength,
        /* [size_is(shapeLength), in] */__int64 * shape,
        /* [in] */__x_ABI_CWindows_CStorage_CStreams_CIBuffer * buffer,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16Bit * * result
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics2Vtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics2
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics2Vtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics2_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics2_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics2_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics2_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics2_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics2_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics2_CreateFromShapeArrayAndDataArray(This,shapeLength,shape,dataLength,data,result) \
    ( (This)->lpVtbl->CreateFromShapeArrayAndDataArray(This,shapeLength,shape,dataLength,data,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics2_CreateFromBuffer(This,shapeLength,shape,buffer,result) \
    ( (This)->lpVtbl->CreateFromBuffer(This,shapeLength,shape,buffer,result) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics2;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorFloat16BitStatics2_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000


/*
 *
 * Class Microsoft.AI.MachineLearning.TensorFloat16Bit
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * RuntimeClass contains static methods.
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorFloat16BitStatics2 interface starting with version 2.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorFloat16BitStatics interface starting with version 1.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.ITensorFloat16Bit ** Default Interface **
 *    Microsoft.AI.MachineLearning.ITensor
 *    Windows.Foundation.IMemoryBuffer
 *
 * Class Threading Model:  Both Single and Multi Threaded Apartment
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorFloat16Bit_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorFloat16Bit_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_TensorFloat16Bit[] = L"Microsoft.AI.MachineLearning.TensorFloat16Bit";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000



/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorUInt8Bit
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorUInt8Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorUInt8Bit:HRESULT GetAsVectorView(Windows.Foundation.Collections.IVectorView`1<UInt8>**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8Bit_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8Bit_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorUInt8Bit[] = L"Microsoft.AI.MachineLearning.ITensorUInt8Bit";
/* [uuid("9156f628-73c4-52b7-931e-0f010c06d5ee"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8Bit * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8Bit * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8Bit * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8Bit * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8Bit * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8Bit * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
HRESULT ( STDMETHODCALLTYPE *GetAsVectorView )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8Bit * This,
        /* [out, retval] */__FIVectorView_1_byte * * result
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8Bit
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8Bit_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8Bit_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8Bit_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8Bit_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8Bit_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8Bit_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8Bit_GetAsVectorView(This,result) \
    ( (This)->lpVtbl->GetAsVectorView(This,result) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8Bit;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8Bit_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorUInt8BitStatics
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorUInt8Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorUInt8BitStatics:HRESULT Create(Microsoft.AI.MachineLearning.TensorUInt8Bit**);HRESULT Create2(Windows.Foundation.Collections.IIterable`1<Int64>*,Microsoft.AI.MachineLearning.TensorUInt8Bit**);HRESULT CreateFromArray(Windows.Foundation.Collections.IIterable`1<Int64>*,UInt32,UInt8*,Microsoft.AI.MachineLearning.TensorUInt8Bit**);HRESULT CreateFromIterable(Windows.Foundation.Collections.IIterable`1<Int64>*,Windows.Foundation.Collections.IIterable`1<UInt8>*,Microsoft.AI.MachineLearning.TensorUInt8Bit**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorUInt8BitStatics[] = L"Microsoft.AI.MachineLearning.ITensorUInt8BitStatics";
/* [uuid("977e8827-fee9-57c3-a7b1-0aa1c3bc6457"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStaticsVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
/* [overload] */HRESULT ( STDMETHODCALLTYPE *Create )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics * This,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8Bit * * result
        );
    /* [overload] */HRESULT ( STDMETHODCALLTYPE *Create2 )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics * This,
        /* [in] */__FIIterable_1___z__zint64 * shape,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8Bit * * result
        );
    HRESULT ( STDMETHODCALLTYPE *CreateFromArray )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics * This,
        /* [in] */__FIIterable_1___z__zint64 * shape,
        /* [in] */unsigned int dataLength,
        /* [size_is(dataLength), in] */byte * data,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8Bit * * result
        );
    HRESULT ( STDMETHODCALLTYPE *CreateFromIterable )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics * This,
        /* [in] */__FIIterable_1___z__zint64 * shape,
        /* [in] */__FIIterable_1_byte * data,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8Bit * * result
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStaticsVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStaticsVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics_Create(This,result) \
    ( (This)->lpVtbl->Create(This,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics_Create2(This,shape,result) \
    ( (This)->lpVtbl->Create2(This,shape,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics_CreateFromArray(This,shape,dataLength,data,result) \
    ( (This)->lpVtbl->CreateFromArray(This,shape,dataLength,data,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics_CreateFromIterable(This,shape,data,result) \
    ( (This)->lpVtbl->CreateFromIterable(This,shape,data,result) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorUInt8BitStatics2
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 2.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorUInt8Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorUInt8BitStatics2:HRESULT CreateFromShapeArrayAndDataArray(UInt32,Int64*,UInt32,UInt8*,Microsoft.AI.MachineLearning.TensorUInt8Bit**);HRESULT CreateFromBuffer(UInt32,Int64*,Windows.Storage.Streams.IBuffer*,Microsoft.AI.MachineLearning.TensorUInt8Bit**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics2_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics2_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorUInt8BitStatics2[] = L"Microsoft.AI.MachineLearning.ITensorUInt8BitStatics2";
/* [uuid("790b9742-c09f-584a-910d-9f5197cbecf4"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics2Vtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics2 * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics2 * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics2 * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics2 * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics2 * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics2 * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
HRESULT ( STDMETHODCALLTYPE *CreateFromShapeArrayAndDataArray )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics2 * This,
        /* [in] */unsigned int shapeLength,
        /* [size_is(shapeLength), in] */__int64 * shape,
        /* [in] */unsigned int dataLength,
        /* [size_is(dataLength), in] */byte * data,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8Bit * * result
        );
    HRESULT ( STDMETHODCALLTYPE *CreateFromBuffer )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics2 * This,
        /* [in] */unsigned int shapeLength,
        /* [size_is(shapeLength), in] */__int64 * shape,
        /* [in] */__x_ABI_CWindows_CStorage_CStreams_CIBuffer * buffer,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8Bit * * result
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics2Vtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics2
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics2Vtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics2_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics2_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics2_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics2_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics2_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics2_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics2_CreateFromShapeArrayAndDataArray(This,shapeLength,shape,dataLength,data,result) \
    ( (This)->lpVtbl->CreateFromShapeArrayAndDataArray(This,shapeLength,shape,dataLength,data,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics2_CreateFromBuffer(This,shapeLength,shape,buffer,result) \
    ( (This)->lpVtbl->CreateFromBuffer(This,shapeLength,shape,buffer,result) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics2;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt8BitStatics2_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000


/*
 *
 * Class Microsoft.AI.MachineLearning.TensorUInt8Bit
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * RuntimeClass contains static methods.
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorUInt8BitStatics2 interface starting with version 2.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorUInt8BitStatics interface starting with version 1.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.ITensorUInt8Bit ** Default Interface **
 *    Microsoft.AI.MachineLearning.ITensor
 *    Windows.Foundation.IMemoryBuffer
 *
 * Class Threading Model:  Both Single and Multi Threaded Apartment
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorUInt8Bit_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorUInt8Bit_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_TensorUInt8Bit[] = L"Microsoft.AI.MachineLearning.TensorUInt8Bit";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000



/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorInt8Bit
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorInt8Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorInt8Bit:HRESULT GetAsVectorView(Windows.Foundation.Collections.IVectorView`1<UInt8>**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8Bit_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8Bit_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorInt8Bit[] = L"Microsoft.AI.MachineLearning.ITensorInt8Bit";
/* [uuid("c7cfa5d8-b290-5310-bf7c-cbf92c9de764"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8Bit * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8Bit * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8Bit * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8Bit * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8Bit * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8Bit * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
HRESULT ( STDMETHODCALLTYPE *GetAsVectorView )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8Bit * This,
        /* [out, retval] */__FIVectorView_1_byte * * result
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8Bit
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8Bit_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8Bit_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8Bit_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8Bit_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8Bit_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8Bit_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8Bit_GetAsVectorView(This,result) \
    ( (This)->lpVtbl->GetAsVectorView(This,result) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8Bit;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8Bit_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorInt8BitStatics
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorInt8Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorInt8BitStatics:HRESULT Create(Microsoft.AI.MachineLearning.TensorInt8Bit**);HRESULT Create2(Windows.Foundation.Collections.IIterable`1<Int64>*,Microsoft.AI.MachineLearning.TensorInt8Bit**);HRESULT CreateFromArray(Windows.Foundation.Collections.IIterable`1<Int64>*,UInt32,UInt8*,Microsoft.AI.MachineLearning.TensorInt8Bit**);HRESULT CreateFromIterable(Windows.Foundation.Collections.IIterable`1<Int64>*,Windows.Foundation.Collections.IIterable`1<UInt8>*,Microsoft.AI.MachineLearning.TensorInt8Bit**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorInt8BitStatics[] = L"Microsoft.AI.MachineLearning.ITensorInt8BitStatics";
/* [uuid("3da5e2b3-4cf6-5d6f-84d3-0b168bd9cba0"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStaticsVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
/* [overload] */HRESULT ( STDMETHODCALLTYPE *Create )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics * This,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8Bit * * result
        );
    /* [overload] */HRESULT ( STDMETHODCALLTYPE *Create2 )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics * This,
        /* [in] */__FIIterable_1___z__zint64 * shape,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8Bit * * result
        );
    HRESULT ( STDMETHODCALLTYPE *CreateFromArray )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics * This,
        /* [in] */__FIIterable_1___z__zint64 * shape,
        /* [in] */unsigned int dataLength,
        /* [size_is(dataLength), in] */BYTE * data,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8Bit * * result
        );
    HRESULT ( STDMETHODCALLTYPE *CreateFromIterable )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics * This,
        /* [in] */__FIIterable_1___z__zint64 * shape,
        /* [in] */__FIIterable_1_byte * data,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8Bit * * result
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStaticsVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStaticsVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics_Create(This,result) \
    ( (This)->lpVtbl->Create(This,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics_Create2(This,shape,result) \
    ( (This)->lpVtbl->Create2(This,shape,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics_CreateFromArray(This,shape,dataLength,data,result) \
    ( (This)->lpVtbl->CreateFromArray(This,shape,dataLength,data,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics_CreateFromIterable(This,shape,data,result) \
    ( (This)->lpVtbl->CreateFromIterable(This,shape,data,result) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorInt8BitStatics2
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 2.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorInt8Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorInt8BitStatics2:HRESULT CreateFromShapeArrayAndDataArray(UInt32,Int64*,UInt32,UInt8*,Microsoft.AI.MachineLearning.TensorInt8Bit**);HRESULT CreateFromBuffer(UInt32,Int64*,Windows.Storage.Streams.IBuffer*,Microsoft.AI.MachineLearning.TensorInt8Bit**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics2_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics2_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorInt8BitStatics2[] = L"Microsoft.AI.MachineLearning.ITensorInt8BitStatics2";
/* [uuid("1cb2caa5-352e-5984-8f43-7e4c611b0e52"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics2Vtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics2 * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics2 * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics2 * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics2 * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics2 * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics2 * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
HRESULT ( STDMETHODCALLTYPE *CreateFromShapeArrayAndDataArray )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics2 * This,
        /* [in] */unsigned int shapeLength,
        /* [size_is(shapeLength), in] */__int64 * shape,
        /* [in] */unsigned int dataLength,
        /* [size_is(dataLength), in] */BYTE * data,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8Bit * * result
        );
    HRESULT ( STDMETHODCALLTYPE *CreateFromBuffer )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics2 * This,
        /* [in] */unsigned int shapeLength,
        /* [size_is(shapeLength), in] */__int64 * shape,
        /* [in] */__x_ABI_CWindows_CStorage_CStreams_CIBuffer * buffer,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8Bit * * result
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics2Vtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics2
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics2Vtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics2_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics2_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics2_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics2_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics2_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics2_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics2_CreateFromShapeArrayAndDataArray(This,shapeLength,shape,dataLength,data,result) \
    ( (This)->lpVtbl->CreateFromShapeArrayAndDataArray(This,shapeLength,shape,dataLength,data,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics2_CreateFromBuffer(This,shapeLength,shape,buffer,result) \
    ( (This)->lpVtbl->CreateFromBuffer(This,shapeLength,shape,buffer,result) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics2;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt8BitStatics2_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000


/*
 *
 * Class Microsoft.AI.MachineLearning.TensorInt8Bit
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * RuntimeClass contains static methods.
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorInt8BitStatics2 interface starting with version 2.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorInt8BitStatics interface starting with version 1.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.ITensorInt8Bit ** Default Interface **
 *    Microsoft.AI.MachineLearning.ITensor
 *    Windows.Foundation.IMemoryBuffer
 *
 * Class Threading Model:  Both Single and Multi Threaded Apartment
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorInt8Bit_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorInt8Bit_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_TensorInt8Bit[] = L"Microsoft.AI.MachineLearning.TensorInt8Bit";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000



/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorUInt16Bit
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorUInt16Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorUInt16Bit:HRESULT GetAsVectorView(Windows.Foundation.Collections.IVectorView`1<UInt16>**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16Bit_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16Bit_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorUInt16Bit[] = L"Microsoft.AI.MachineLearning.ITensorUInt16Bit";
/* [uuid("33ea36fb-5129-5c13-b591-f407a6bfe637"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16Bit * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16Bit * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16Bit * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16Bit * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16Bit * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16Bit * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
HRESULT ( STDMETHODCALLTYPE *GetAsVectorView )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16Bit * This,
        /* [out, retval] */__FIVectorView_1_UINT16 * * result
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16Bit
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16Bit_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16Bit_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16Bit_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16Bit_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16Bit_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16Bit_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16Bit_GetAsVectorView(This,result) \
    ( (This)->lpVtbl->GetAsVectorView(This,result) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16Bit;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16Bit_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorUInt16BitStatics
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorUInt16Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorUInt16BitStatics:HRESULT Create(Microsoft.AI.MachineLearning.TensorUInt16Bit**);HRESULT Create2(Windows.Foundation.Collections.IIterable`1<Int64>*,Microsoft.AI.MachineLearning.TensorUInt16Bit**);HRESULT CreateFromArray(Windows.Foundation.Collections.IIterable`1<Int64>*,UInt32,UInt16*,Microsoft.AI.MachineLearning.TensorUInt16Bit**);HRESULT CreateFromIterable(Windows.Foundation.Collections.IIterable`1<Int64>*,Windows.Foundation.Collections.IIterable`1<UInt16>*,Microsoft.AI.MachineLearning.TensorUInt16Bit**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorUInt16BitStatics[] = L"Microsoft.AI.MachineLearning.ITensorUInt16BitStatics";
/* [uuid("93b94404-8dfc-592b-b51b-fbcf79261401"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStaticsVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
/* [overload] */HRESULT ( STDMETHODCALLTYPE *Create )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics * This,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16Bit * * result
        );
    /* [overload] */HRESULT ( STDMETHODCALLTYPE *Create2 )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics * This,
        /* [in] */__FIIterable_1___z__zint64 * shape,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16Bit * * result
        );
    HRESULT ( STDMETHODCALLTYPE *CreateFromArray )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics * This,
        /* [in] */__FIIterable_1___z__zint64 * shape,
        /* [in] */unsigned int dataLength,
        /* [size_is(dataLength), in] */unsigned short * data,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16Bit * * result
        );
    HRESULT ( STDMETHODCALLTYPE *CreateFromIterable )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics * This,
        /* [in] */__FIIterable_1___z__zint64 * shape,
        /* [in] */__FIIterable_1_UINT16 * data,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16Bit * * result
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStaticsVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStaticsVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics_Create(This,result) \
    ( (This)->lpVtbl->Create(This,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics_Create2(This,shape,result) \
    ( (This)->lpVtbl->Create2(This,shape,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics_CreateFromArray(This,shape,dataLength,data,result) \
    ( (This)->lpVtbl->CreateFromArray(This,shape,dataLength,data,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics_CreateFromIterable(This,shape,data,result) \
    ( (This)->lpVtbl->CreateFromIterable(This,shape,data,result) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorUInt16BitStatics2
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 2.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorUInt16Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorUInt16BitStatics2:HRESULT CreateFromShapeArrayAndDataArray(UInt32,Int64*,UInt32,UInt16*,Microsoft.AI.MachineLearning.TensorUInt16Bit**);HRESULT CreateFromBuffer(UInt32,Int64*,Windows.Storage.Streams.IBuffer*,Microsoft.AI.MachineLearning.TensorUInt16Bit**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics2_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics2_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorUInt16BitStatics2[] = L"Microsoft.AI.MachineLearning.ITensorUInt16BitStatics2";
/* [uuid("8cc5af95-d59e-56de-aec9-c43cf9a7271f"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics2Vtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics2 * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics2 * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics2 * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics2 * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics2 * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics2 * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
HRESULT ( STDMETHODCALLTYPE *CreateFromShapeArrayAndDataArray )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics2 * This,
        /* [in] */unsigned int shapeLength,
        /* [size_is(shapeLength), in] */__int64 * shape,
        /* [in] */unsigned int dataLength,
        /* [size_is(dataLength), in] */unsigned short * data,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16Bit * * result
        );
    HRESULT ( STDMETHODCALLTYPE *CreateFromBuffer )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics2 * This,
        /* [in] */unsigned int shapeLength,
        /* [size_is(shapeLength), in] */__int64 * shape,
        /* [in] */__x_ABI_CWindows_CStorage_CStreams_CIBuffer * buffer,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16Bit * * result
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics2Vtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics2
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics2Vtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics2_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics2_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics2_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics2_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics2_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics2_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics2_CreateFromShapeArrayAndDataArray(This,shapeLength,shape,dataLength,data,result) \
    ( (This)->lpVtbl->CreateFromShapeArrayAndDataArray(This,shapeLength,shape,dataLength,data,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics2_CreateFromBuffer(This,shapeLength,shape,buffer,result) \
    ( (This)->lpVtbl->CreateFromBuffer(This,shapeLength,shape,buffer,result) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics2;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt16BitStatics2_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000


/*
 *
 * Class Microsoft.AI.MachineLearning.TensorUInt16Bit
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * RuntimeClass contains static methods.
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorUInt16BitStatics2 interface starting with version 2.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorUInt16BitStatics interface starting with version 1.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.ITensorUInt16Bit ** Default Interface **
 *    Microsoft.AI.MachineLearning.ITensor
 *    Windows.Foundation.IMemoryBuffer
 *
 * Class Threading Model:  Both Single and Multi Threaded Apartment
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorUInt16Bit_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorUInt16Bit_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_TensorUInt16Bit[] = L"Microsoft.AI.MachineLearning.TensorUInt16Bit";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000



/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorInt16Bit
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorInt16Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorInt16Bit:HRESULT GetAsVectorView(Windows.Foundation.Collections.IVectorView`1<Int16>**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16Bit_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16Bit_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorInt16Bit[] = L"Microsoft.AI.MachineLearning.ITensorInt16Bit";
/* [uuid("a757726f-ddf3-5d6f-ab2e-17e5aad63088"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16Bit * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16Bit * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16Bit * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16Bit * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16Bit * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16Bit * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
HRESULT ( STDMETHODCALLTYPE *GetAsVectorView )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16Bit * This,
        /* [out, retval] */__FIVectorView_1_short * * result
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16Bit
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16Bit_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16Bit_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16Bit_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16Bit_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16Bit_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16Bit_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16Bit_GetAsVectorView(This,result) \
    ( (This)->lpVtbl->GetAsVectorView(This,result) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16Bit;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16Bit_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorInt16BitStatics
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorInt16Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorInt16BitStatics:HRESULT Create(Microsoft.AI.MachineLearning.TensorInt16Bit**);HRESULT Create2(Windows.Foundation.Collections.IIterable`1<Int64>*,Microsoft.AI.MachineLearning.TensorInt16Bit**);HRESULT CreateFromArray(Windows.Foundation.Collections.IIterable`1<Int64>*,UInt32,Int16*,Microsoft.AI.MachineLearning.TensorInt16Bit**);HRESULT CreateFromIterable(Windows.Foundation.Collections.IIterable`1<Int64>*,Windows.Foundation.Collections.IIterable`1<Int16>*,Microsoft.AI.MachineLearning.TensorInt16Bit**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorInt16BitStatics[] = L"Microsoft.AI.MachineLearning.ITensorInt16BitStatics";
/* [uuid("5cabb6d2-4ee9-5967-ae6f-59d4a1c6d846"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStaticsVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
/* [overload] */HRESULT ( STDMETHODCALLTYPE *Create )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics * This,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16Bit * * result
        );
    /* [overload] */HRESULT ( STDMETHODCALLTYPE *Create2 )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics * This,
        /* [in] */__FIIterable_1___z__zint64 * shape,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16Bit * * result
        );
    HRESULT ( STDMETHODCALLTYPE *CreateFromArray )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics * This,
        /* [in] */__FIIterable_1___z__zint64 * shape,
        /* [in] */unsigned int dataLength,
        /* [size_is(dataLength), in] */short * data,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16Bit * * result
        );
    HRESULT ( STDMETHODCALLTYPE *CreateFromIterable )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics * This,
        /* [in] */__FIIterable_1___z__zint64 * shape,
        /* [in] */__FIIterable_1_short * data,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16Bit * * result
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStaticsVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStaticsVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics_Create(This,result) \
    ( (This)->lpVtbl->Create(This,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics_Create2(This,shape,result) \
    ( (This)->lpVtbl->Create2(This,shape,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics_CreateFromArray(This,shape,dataLength,data,result) \
    ( (This)->lpVtbl->CreateFromArray(This,shape,dataLength,data,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics_CreateFromIterable(This,shape,data,result) \
    ( (This)->lpVtbl->CreateFromIterable(This,shape,data,result) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorInt16BitStatics2
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 2.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorInt16Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorInt16BitStatics2:HRESULT CreateFromShapeArrayAndDataArray(UInt32,Int64*,UInt32,Int16*,Microsoft.AI.MachineLearning.TensorInt16Bit**);HRESULT CreateFromBuffer(UInt32,Int64*,Windows.Storage.Streams.IBuffer*,Microsoft.AI.MachineLearning.TensorInt16Bit**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics2_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics2_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorInt16BitStatics2[] = L"Microsoft.AI.MachineLearning.ITensorInt16BitStatics2";
/* [uuid("91eee292-5dbc-542f-a3d4-c675e6b7c31d"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics2Vtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics2 * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics2 * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics2 * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics2 * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics2 * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics2 * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
HRESULT ( STDMETHODCALLTYPE *CreateFromShapeArrayAndDataArray )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics2 * This,
        /* [in] */unsigned int shapeLength,
        /* [size_is(shapeLength), in] */__int64 * shape,
        /* [in] */unsigned int dataLength,
        /* [size_is(dataLength), in] */short * data,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16Bit * * result
        );
    HRESULT ( STDMETHODCALLTYPE *CreateFromBuffer )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics2 * This,
        /* [in] */unsigned int shapeLength,
        /* [size_is(shapeLength), in] */__int64 * shape,
        /* [in] */__x_ABI_CWindows_CStorage_CStreams_CIBuffer * buffer,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16Bit * * result
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics2Vtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics2
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics2Vtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics2_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics2_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics2_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics2_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics2_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics2_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics2_CreateFromShapeArrayAndDataArray(This,shapeLength,shape,dataLength,data,result) \
    ( (This)->lpVtbl->CreateFromShapeArrayAndDataArray(This,shapeLength,shape,dataLength,data,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics2_CreateFromBuffer(This,shapeLength,shape,buffer,result) \
    ( (This)->lpVtbl->CreateFromBuffer(This,shapeLength,shape,buffer,result) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics2;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt16BitStatics2_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000


/*
 *
 * Class Microsoft.AI.MachineLearning.TensorInt16Bit
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * RuntimeClass contains static methods.
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorInt16BitStatics2 interface starting with version 2.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorInt16BitStatics interface starting with version 1.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.ITensorInt16Bit ** Default Interface **
 *    Microsoft.AI.MachineLearning.ITensor
 *    Windows.Foundation.IMemoryBuffer
 *
 * Class Threading Model:  Both Single and Multi Threaded Apartment
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorInt16Bit_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorInt16Bit_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_TensorInt16Bit[] = L"Microsoft.AI.MachineLearning.TensorInt16Bit";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000



/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorUInt32Bit
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorUInt32Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorUInt32Bit:HRESULT GetAsVectorView(Windows.Foundation.Collections.IVectorView`1<UInt32>**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32Bit_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32Bit_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorUInt32Bit[] = L"Microsoft.AI.MachineLearning.ITensorUInt32Bit";
/* [uuid("4453ec13-4a94-5d72-bd57-ac306c74d620"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32Bit * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32Bit * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32Bit * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32Bit * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32Bit * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32Bit * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
HRESULT ( STDMETHODCALLTYPE *GetAsVectorView )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32Bit * This,
        /* [out, retval] */__FIVectorView_1_UINT32 * * result
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32Bit
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32Bit_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32Bit_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32Bit_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32Bit_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32Bit_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32Bit_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32Bit_GetAsVectorView(This,result) \
    ( (This)->lpVtbl->GetAsVectorView(This,result) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32Bit;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32Bit_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorUInt32BitStatics
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorUInt32Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorUInt32BitStatics:HRESULT Create(Microsoft.AI.MachineLearning.TensorUInt32Bit**);HRESULT Create2(Windows.Foundation.Collections.IIterable`1<Int64>*,Microsoft.AI.MachineLearning.TensorUInt32Bit**);HRESULT CreateFromArray(Windows.Foundation.Collections.IIterable`1<Int64>*,UInt32,UInt32*,Microsoft.AI.MachineLearning.TensorUInt32Bit**);HRESULT CreateFromIterable(Windows.Foundation.Collections.IIterable`1<Int64>*,Windows.Foundation.Collections.IIterable`1<UInt32>*,Microsoft.AI.MachineLearning.TensorUInt32Bit**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorUInt32BitStatics[] = L"Microsoft.AI.MachineLearning.ITensorUInt32BitStatics";
/* [uuid("8b23c75c-3a66-5336-b509-45350f26bf09"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStaticsVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
/* [overload] */HRESULT ( STDMETHODCALLTYPE *Create )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics * This,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32Bit * * result
        );
    /* [overload] */HRESULT ( STDMETHODCALLTYPE *Create2 )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics * This,
        /* [in] */__FIIterable_1___z__zint64 * shape,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32Bit * * result
        );
    HRESULT ( STDMETHODCALLTYPE *CreateFromArray )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics * This,
        /* [in] */__FIIterable_1___z__zint64 * shape,
        /* [in] */unsigned int dataLength,
        /* [size_is(dataLength), in] */unsigned int * data,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32Bit * * result
        );
    HRESULT ( STDMETHODCALLTYPE *CreateFromIterable )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics * This,
        /* [in] */__FIIterable_1___z__zint64 * shape,
        /* [in] */__FIIterable_1_UINT32 * data,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32Bit * * result
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStaticsVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStaticsVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics_Create(This,result) \
    ( (This)->lpVtbl->Create(This,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics_Create2(This,shape,result) \
    ( (This)->lpVtbl->Create2(This,shape,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics_CreateFromArray(This,shape,dataLength,data,result) \
    ( (This)->lpVtbl->CreateFromArray(This,shape,dataLength,data,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics_CreateFromIterable(This,shape,data,result) \
    ( (This)->lpVtbl->CreateFromIterable(This,shape,data,result) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorUInt32BitStatics2
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 2.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorUInt32Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorUInt32BitStatics2:HRESULT CreateFromShapeArrayAndDataArray(UInt32,Int64*,UInt32,UInt32*,Microsoft.AI.MachineLearning.TensorUInt32Bit**);HRESULT CreateFromBuffer(UInt32,Int64*,Windows.Storage.Streams.IBuffer*,Microsoft.AI.MachineLearning.TensorUInt32Bit**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics2_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics2_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorUInt32BitStatics2[] = L"Microsoft.AI.MachineLearning.ITensorUInt32BitStatics2";
/* [uuid("b53b05b9-c812-5da6-bfa8-aaff5bfdae79"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics2Vtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics2 * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics2 * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics2 * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics2 * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics2 * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics2 * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
HRESULT ( STDMETHODCALLTYPE *CreateFromShapeArrayAndDataArray )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics2 * This,
        /* [in] */unsigned int shapeLength,
        /* [size_is(shapeLength), in] */__int64 * shape,
        /* [in] */unsigned int dataLength,
        /* [size_is(dataLength), in] */unsigned int * data,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32Bit * * result
        );
    HRESULT ( STDMETHODCALLTYPE *CreateFromBuffer )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics2 * This,
        /* [in] */unsigned int shapeLength,
        /* [size_is(shapeLength), in] */__int64 * shape,
        /* [in] */__x_ABI_CWindows_CStorage_CStreams_CIBuffer * buffer,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32Bit * * result
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics2Vtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics2
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics2Vtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics2_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics2_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics2_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics2_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics2_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics2_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics2_CreateFromShapeArrayAndDataArray(This,shapeLength,shape,dataLength,data,result) \
    ( (This)->lpVtbl->CreateFromShapeArrayAndDataArray(This,shapeLength,shape,dataLength,data,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics2_CreateFromBuffer(This,shapeLength,shape,buffer,result) \
    ( (This)->lpVtbl->CreateFromBuffer(This,shapeLength,shape,buffer,result) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics2;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt32BitStatics2_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000


/*
 *
 * Class Microsoft.AI.MachineLearning.TensorUInt32Bit
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * RuntimeClass contains static methods.
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorUInt32BitStatics2 interface starting with version 2.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorUInt32BitStatics interface starting with version 1.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.ITensorUInt32Bit ** Default Interface **
 *    Microsoft.AI.MachineLearning.ITensor
 *    Windows.Foundation.IMemoryBuffer
 *
 * Class Threading Model:  Both Single and Multi Threaded Apartment
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorUInt32Bit_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorUInt32Bit_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_TensorUInt32Bit[] = L"Microsoft.AI.MachineLearning.TensorUInt32Bit";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000



/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorInt32Bit
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorInt32Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorInt32Bit:HRESULT GetAsVectorView(Windows.Foundation.Collections.IVectorView`1<Int32>**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32Bit_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32Bit_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorInt32Bit[] = L"Microsoft.AI.MachineLearning.ITensorInt32Bit";
/* [uuid("0daf5ea0-661f-5bc6-b210-2960138c688e"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32Bit * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32Bit * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32Bit * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32Bit * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32Bit * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32Bit * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
HRESULT ( STDMETHODCALLTYPE *GetAsVectorView )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32Bit * This,
        /* [out, retval] */__FIVectorView_1_int * * result
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32Bit
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32Bit_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32Bit_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32Bit_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32Bit_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32Bit_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32Bit_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32Bit_GetAsVectorView(This,result) \
    ( (This)->lpVtbl->GetAsVectorView(This,result) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32Bit;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32Bit_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorInt32BitStatics
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorInt32Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorInt32BitStatics:HRESULT Create(Microsoft.AI.MachineLearning.TensorInt32Bit**);HRESULT Create2(Windows.Foundation.Collections.IIterable`1<Int64>*,Microsoft.AI.MachineLearning.TensorInt32Bit**);HRESULT CreateFromArray(Windows.Foundation.Collections.IIterable`1<Int64>*,UInt32,Int32*,Microsoft.AI.MachineLearning.TensorInt32Bit**);HRESULT CreateFromIterable(Windows.Foundation.Collections.IIterable`1<Int64>*,Windows.Foundation.Collections.IIterable`1<Int32>*,Microsoft.AI.MachineLearning.TensorInt32Bit**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorInt32BitStatics[] = L"Microsoft.AI.MachineLearning.ITensorInt32BitStatics";
/* [uuid("de454982-a86e-54aa-b88d-e34b66b3c001"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStaticsVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
/* [overload] */HRESULT ( STDMETHODCALLTYPE *Create )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics * This,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32Bit * * result
        );
    /* [overload] */HRESULT ( STDMETHODCALLTYPE *Create2 )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics * This,
        /* [in] */__FIIterable_1___z__zint64 * shape,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32Bit * * result
        );
    HRESULT ( STDMETHODCALLTYPE *CreateFromArray )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics * This,
        /* [in] */__FIIterable_1___z__zint64 * shape,
        /* [in] */unsigned int dataLength,
        /* [size_is(dataLength), in] */int * data,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32Bit * * result
        );
    HRESULT ( STDMETHODCALLTYPE *CreateFromIterable )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics * This,
        /* [in] */__FIIterable_1___z__zint64 * shape,
        /* [in] */__FIIterable_1_int * data,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32Bit * * result
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStaticsVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStaticsVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics_Create(This,result) \
    ( (This)->lpVtbl->Create(This,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics_Create2(This,shape,result) \
    ( (This)->lpVtbl->Create2(This,shape,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics_CreateFromArray(This,shape,dataLength,data,result) \
    ( (This)->lpVtbl->CreateFromArray(This,shape,dataLength,data,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics_CreateFromIterable(This,shape,data,result) \
    ( (This)->lpVtbl->CreateFromIterable(This,shape,data,result) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorInt32BitStatics2
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 2.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorInt32Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorInt32BitStatics2:HRESULT CreateFromShapeArrayAndDataArray(UInt32,Int64*,UInt32,Int32*,Microsoft.AI.MachineLearning.TensorInt32Bit**);HRESULT CreateFromBuffer(UInt32,Int64*,Windows.Storage.Streams.IBuffer*,Microsoft.AI.MachineLearning.TensorInt32Bit**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics2_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics2_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorInt32BitStatics2[] = L"Microsoft.AI.MachineLearning.ITensorInt32BitStatics2";
/* [uuid("a7924918-4afe-51d4-9f84-5dee7bdbe500"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics2Vtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics2 * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics2 * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics2 * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics2 * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics2 * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics2 * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
HRESULT ( STDMETHODCALLTYPE *CreateFromShapeArrayAndDataArray )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics2 * This,
        /* [in] */unsigned int shapeLength,
        /* [size_is(shapeLength), in] */__int64 * shape,
        /* [in] */unsigned int dataLength,
        /* [size_is(dataLength), in] */int * data,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32Bit * * result
        );
    HRESULT ( STDMETHODCALLTYPE *CreateFromBuffer )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics2 * This,
        /* [in] */unsigned int shapeLength,
        /* [size_is(shapeLength), in] */__int64 * shape,
        /* [in] */__x_ABI_CWindows_CStorage_CStreams_CIBuffer * buffer,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32Bit * * result
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics2Vtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics2
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics2Vtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics2_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics2_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics2_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics2_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics2_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics2_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics2_CreateFromShapeArrayAndDataArray(This,shapeLength,shape,dataLength,data,result) \
    ( (This)->lpVtbl->CreateFromShapeArrayAndDataArray(This,shapeLength,shape,dataLength,data,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics2_CreateFromBuffer(This,shapeLength,shape,buffer,result) \
    ( (This)->lpVtbl->CreateFromBuffer(This,shapeLength,shape,buffer,result) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics2;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt32BitStatics2_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000


/*
 *
 * Class Microsoft.AI.MachineLearning.TensorInt32Bit
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * RuntimeClass contains static methods.
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorInt32BitStatics2 interface starting with version 2.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorInt32BitStatics interface starting with version 1.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.ITensorInt32Bit ** Default Interface **
 *    Microsoft.AI.MachineLearning.ITensor
 *    Windows.Foundation.IMemoryBuffer
 *
 * Class Threading Model:  Both Single and Multi Threaded Apartment
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorInt32Bit_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorInt32Bit_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_TensorInt32Bit[] = L"Microsoft.AI.MachineLearning.TensorInt32Bit";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000



/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorUInt64Bit
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorUInt64Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorUInt64Bit:HRESULT GetAsVectorView(Windows.Foundation.Collections.IVectorView`1<UInt64>**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64Bit_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64Bit_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorUInt64Bit[] = L"Microsoft.AI.MachineLearning.ITensorUInt64Bit";
/* [uuid("c04b9e9b-fc04-58f4-b39c-c04d7f5777ea"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64Bit * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64Bit * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64Bit * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64Bit * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64Bit * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64Bit * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
HRESULT ( STDMETHODCALLTYPE *GetAsVectorView )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64Bit * This,
        /* [out, retval] */__FIVectorView_1_UINT64 * * result
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64Bit
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64Bit_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64Bit_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64Bit_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64Bit_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64Bit_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64Bit_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64Bit_GetAsVectorView(This,result) \
    ( (This)->lpVtbl->GetAsVectorView(This,result) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64Bit;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64Bit_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorUInt64BitStatics
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorUInt64Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorUInt64BitStatics:HRESULT Create(Microsoft.AI.MachineLearning.TensorUInt64Bit**);HRESULT Create2(Windows.Foundation.Collections.IIterable`1<Int64>*,Microsoft.AI.MachineLearning.TensorUInt64Bit**);HRESULT CreateFromArray(Windows.Foundation.Collections.IIterable`1<Int64>*,UInt32,UInt64*,Microsoft.AI.MachineLearning.TensorUInt64Bit**);HRESULT CreateFromIterable(Windows.Foundation.Collections.IIterable`1<Int64>*,Windows.Foundation.Collections.IIterable`1<UInt64>*,Microsoft.AI.MachineLearning.TensorUInt64Bit**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorUInt64BitStatics[] = L"Microsoft.AI.MachineLearning.ITensorUInt64BitStatics";
/* [uuid("0b04cac1-75b4-5e7f-b211-a56ddb3b5f47"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStaticsVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
/* [overload] */HRESULT ( STDMETHODCALLTYPE *Create )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics * This,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64Bit * * result
        );
    /* [overload] */HRESULT ( STDMETHODCALLTYPE *Create2 )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics * This,
        /* [in] */__FIIterable_1___z__zint64 * shape,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64Bit * * result
        );
    HRESULT ( STDMETHODCALLTYPE *CreateFromArray )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics * This,
        /* [in] */__FIIterable_1___z__zint64 * shape,
        /* [in] */unsigned int dataLength,
        /* [size_is(dataLength), in] */unsigned __int64 * data,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64Bit * * result
        );
    HRESULT ( STDMETHODCALLTYPE *CreateFromIterable )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics * This,
        /* [in] */__FIIterable_1___z__zint64 * shape,
        /* [in] */__FIIterable_1_UINT64 * data,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64Bit * * result
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStaticsVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStaticsVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics_Create(This,result) \
    ( (This)->lpVtbl->Create(This,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics_Create2(This,shape,result) \
    ( (This)->lpVtbl->Create2(This,shape,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics_CreateFromArray(This,shape,dataLength,data,result) \
    ( (This)->lpVtbl->CreateFromArray(This,shape,dataLength,data,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics_CreateFromIterable(This,shape,data,result) \
    ( (This)->lpVtbl->CreateFromIterable(This,shape,data,result) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorUInt64BitStatics2
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 2.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorUInt64Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorUInt64BitStatics2:HRESULT CreateFromShapeArrayAndDataArray(UInt32,Int64*,UInt32,UInt64*,Microsoft.AI.MachineLearning.TensorUInt64Bit**);HRESULT CreateFromBuffer(UInt32,Int64*,Windows.Storage.Streams.IBuffer*,Microsoft.AI.MachineLearning.TensorUInt64Bit**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics2_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics2_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorUInt64BitStatics2[] = L"Microsoft.AI.MachineLearning.ITensorUInt64BitStatics2";
/* [uuid("d3e5a1ed-c9e7-555a-8b9f-a0773b447ae8"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics2Vtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics2 * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics2 * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics2 * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics2 * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics2 * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics2 * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
HRESULT ( STDMETHODCALLTYPE *CreateFromShapeArrayAndDataArray )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics2 * This,
        /* [in] */unsigned int shapeLength,
        /* [size_is(shapeLength), in] */__int64 * shape,
        /* [in] */unsigned int dataLength,
        /* [size_is(dataLength), in] */unsigned __int64 * data,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64Bit * * result
        );
    HRESULT ( STDMETHODCALLTYPE *CreateFromBuffer )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics2 * This,
        /* [in] */unsigned int shapeLength,
        /* [size_is(shapeLength), in] */__int64 * shape,
        /* [in] */__x_ABI_CWindows_CStorage_CStreams_CIBuffer * buffer,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64Bit * * result
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics2Vtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics2
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics2Vtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics2_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics2_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics2_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics2_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics2_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics2_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics2_CreateFromShapeArrayAndDataArray(This,shapeLength,shape,dataLength,data,result) \
    ( (This)->lpVtbl->CreateFromShapeArrayAndDataArray(This,shapeLength,shape,dataLength,data,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics2_CreateFromBuffer(This,shapeLength,shape,buffer,result) \
    ( (This)->lpVtbl->CreateFromBuffer(This,shapeLength,shape,buffer,result) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics2;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorUInt64BitStatics2_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000


/*
 *
 * Class Microsoft.AI.MachineLearning.TensorUInt64Bit
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * RuntimeClass contains static methods.
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorUInt64BitStatics2 interface starting with version 2.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorUInt64BitStatics interface starting with version 1.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.ITensorUInt64Bit ** Default Interface **
 *    Microsoft.AI.MachineLearning.ITensor
 *    Windows.Foundation.IMemoryBuffer
 *
 * Class Threading Model:  Both Single and Multi Threaded Apartment
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorUInt64Bit_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorUInt64Bit_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_TensorUInt64Bit[] = L"Microsoft.AI.MachineLearning.TensorUInt64Bit";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000



/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorInt64Bit
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorInt64Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorInt64Bit:HRESULT GetAsVectorView(Windows.Foundation.Collections.IVectorView`1<Int64>**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64Bit_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64Bit_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorInt64Bit[] = L"Microsoft.AI.MachineLearning.ITensorInt64Bit";
/* [uuid("330157f0-a3c0-5b99-b593-3bac3103e9b3"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64Bit * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64Bit * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64Bit * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64Bit * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64Bit * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64Bit * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
HRESULT ( STDMETHODCALLTYPE *GetAsVectorView )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64Bit * This,
        /* [out, retval] */__FIVectorView_1___z__zint64 * * result
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64Bit
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64Bit_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64Bit_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64Bit_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64Bit_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64Bit_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64Bit_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64Bit_GetAsVectorView(This,result) \
    ( (This)->lpVtbl->GetAsVectorView(This,result) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64Bit;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64Bit_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorInt64BitStatics
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorInt64Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorInt64BitStatics:HRESULT Create(Microsoft.AI.MachineLearning.TensorInt64Bit**);HRESULT Create2(Windows.Foundation.Collections.IIterable`1<Int64>*,Microsoft.AI.MachineLearning.TensorInt64Bit**);HRESULT CreateFromArray(Windows.Foundation.Collections.IIterable`1<Int64>*,UInt32,Int64*,Microsoft.AI.MachineLearning.TensorInt64Bit**);HRESULT CreateFromIterable(Windows.Foundation.Collections.IIterable`1<Int64>*,Windows.Foundation.Collections.IIterable`1<Int64>*,Microsoft.AI.MachineLearning.TensorInt64Bit**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorInt64BitStatics[] = L"Microsoft.AI.MachineLearning.ITensorInt64BitStatics";
/* [uuid("d3aff765-eb3c-5e0f-959b-16c66ecfa5f2"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStaticsVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
/* [overload] */HRESULT ( STDMETHODCALLTYPE *Create )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics * This,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64Bit * * result
        );
    /* [overload] */HRESULT ( STDMETHODCALLTYPE *Create2 )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics * This,
        /* [in] */__FIIterable_1___z__zint64 * shape,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64Bit * * result
        );
    HRESULT ( STDMETHODCALLTYPE *CreateFromArray )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics * This,
        /* [in] */__FIIterable_1___z__zint64 * shape,
        /* [in] */unsigned int dataLength,
        /* [size_is(dataLength), in] */__int64 * data,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64Bit * * result
        );
    HRESULT ( STDMETHODCALLTYPE *CreateFromIterable )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics * This,
        /* [in] */__FIIterable_1___z__zint64 * shape,
        /* [in] */__FIIterable_1___z__zint64 * data,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64Bit * * result
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStaticsVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStaticsVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics_Create(This,result) \
    ( (This)->lpVtbl->Create(This,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics_Create2(This,shape,result) \
    ( (This)->lpVtbl->Create2(This,shape,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics_CreateFromArray(This,shape,dataLength,data,result) \
    ( (This)->lpVtbl->CreateFromArray(This,shape,dataLength,data,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics_CreateFromIterable(This,shape,data,result) \
    ( (This)->lpVtbl->CreateFromIterable(This,shape,data,result) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorInt64BitStatics2
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 2.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorInt64Bit
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorInt64BitStatics2:HRESULT CreateFromShapeArrayAndDataArray(UInt32,Int64*,UInt32,Int64*,Microsoft.AI.MachineLearning.TensorInt64Bit**);HRESULT CreateFromBuffer(UInt32,Int64*,Windows.Storage.Streams.IBuffer*,Microsoft.AI.MachineLearning.TensorInt64Bit**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics2_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics2_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorInt64BitStatics2[] = L"Microsoft.AI.MachineLearning.ITensorInt64BitStatics2";
/* [uuid("bf724240-4b4b-5434-9a1a-5f274ec542f8"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics2Vtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics2 * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics2 * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics2 * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics2 * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics2 * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics2 * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
HRESULT ( STDMETHODCALLTYPE *CreateFromShapeArrayAndDataArray )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics2 * This,
        /* [in] */unsigned int shapeLength,
        /* [size_is(shapeLength), in] */__int64 * shape,
        /* [in] */unsigned int dataLength,
        /* [size_is(dataLength), in] */__int64 * data,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64Bit * * result
        );
    HRESULT ( STDMETHODCALLTYPE *CreateFromBuffer )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics2 * This,
        /* [in] */unsigned int shapeLength,
        /* [size_is(shapeLength), in] */__int64 * shape,
        /* [in] */__x_ABI_CWindows_CStorage_CStreams_CIBuffer * buffer,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64Bit * * result
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics2Vtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics2
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics2Vtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics2_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics2_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics2_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics2_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics2_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics2_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics2_CreateFromShapeArrayAndDataArray(This,shapeLength,shape,dataLength,data,result) \
    ( (This)->lpVtbl->CreateFromShapeArrayAndDataArray(This,shapeLength,shape,dataLength,data,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics2_CreateFromBuffer(This,shapeLength,shape,buffer,result) \
    ( (This)->lpVtbl->CreateFromBuffer(This,shapeLength,shape,buffer,result) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics2;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorInt64BitStatics2_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000


/*
 *
 * Class Microsoft.AI.MachineLearning.TensorInt64Bit
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * RuntimeClass contains static methods.
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorInt64BitStatics2 interface starting with version 2.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorInt64BitStatics interface starting with version 1.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.ITensorInt64Bit ** Default Interface **
 *    Microsoft.AI.MachineLearning.ITensor
 *    Windows.Foundation.IMemoryBuffer
 *
 * Class Threading Model:  Both Single and Multi Threaded Apartment
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorInt64Bit_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorInt64Bit_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_TensorInt64Bit[] = L"Microsoft.AI.MachineLearning.TensorInt64Bit";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000



/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorBoolean
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorBoolean
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorBoolean:HRESULT GetAsVectorView(Windows.Foundation.Collections.IVectorView`1<Boolean>**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBoolean_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBoolean_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorBoolean[] = L"Microsoft.AI.MachineLearning.ITensorBoolean";
/* [uuid("5e234751-afc1-547b-a05f-3be1401d7f6b"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBoolean * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBoolean * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBoolean * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBoolean * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBoolean * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBoolean * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
HRESULT ( STDMETHODCALLTYPE *GetAsVectorView )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBoolean * This,
        /* [out, retval] */__FIVectorView_1_boolean * * result
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBoolean
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBoolean_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBoolean_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBoolean_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBoolean_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBoolean_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBoolean_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBoolean_GetAsVectorView(This,result) \
    ( (This)->lpVtbl->GetAsVectorView(This,result) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBoolean;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBoolean_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorBooleanStatics
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorBoolean
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorBooleanStatics:HRESULT Create(Microsoft.AI.MachineLearning.TensorBoolean**);HRESULT Create2(Windows.Foundation.Collections.IIterable`1<Int64>*,Microsoft.AI.MachineLearning.TensorBoolean**);HRESULT CreateFromArray(Windows.Foundation.Collections.IIterable`1<Int64>*,UInt32,Boolean*,Microsoft.AI.MachineLearning.TensorBoolean**);HRESULT CreateFromIterable(Windows.Foundation.Collections.IIterable`1<Int64>*,Windows.Foundation.Collections.IIterable`1<Boolean>*,Microsoft.AI.MachineLearning.TensorBoolean**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorBooleanStatics[] = L"Microsoft.AI.MachineLearning.ITensorBooleanStatics";
/* [uuid("d9f16dbc-e102-587d-bacb-f31c98da0bb3"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStaticsVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
/* [overload] */HRESULT ( STDMETHODCALLTYPE *Create )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics * This,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBoolean * * result
        );
    /* [overload] */HRESULT ( STDMETHODCALLTYPE *Create2 )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics * This,
        /* [in] */__FIIterable_1___z__zint64 * shape,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBoolean * * result
        );
    HRESULT ( STDMETHODCALLTYPE *CreateFromArray )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics * This,
        /* [in] */__FIIterable_1___z__zint64 * shape,
        /* [in] */unsigned int dataLength,
        /* [size_is(dataLength), in] */boolean * data,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBoolean * * result
        );
    HRESULT ( STDMETHODCALLTYPE *CreateFromIterable )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics * This,
        /* [in] */__FIIterable_1___z__zint64 * shape,
        /* [in] */__FIIterable_1_boolean * data,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBoolean * * result
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStaticsVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStaticsVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics_Create(This,result) \
    ( (This)->lpVtbl->Create(This,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics_Create2(This,shape,result) \
    ( (This)->lpVtbl->Create2(This,shape,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics_CreateFromArray(This,shape,dataLength,data,result) \
    ( (This)->lpVtbl->CreateFromArray(This,shape,dataLength,data,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics_CreateFromIterable(This,shape,data,result) \
    ( (This)->lpVtbl->CreateFromIterable(This,shape,data,result) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorBooleanStatics2
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 2.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorBoolean
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorBooleanStatics2:HRESULT CreateFromShapeArrayAndDataArray(UInt32,Int64*,UInt32,Boolean*,Microsoft.AI.MachineLearning.TensorBoolean**);HRESULT CreateFromBuffer(UInt32,Int64*,Windows.Storage.Streams.IBuffer*,Microsoft.AI.MachineLearning.TensorBoolean**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics2_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics2_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorBooleanStatics2[] = L"Microsoft.AI.MachineLearning.ITensorBooleanStatics2";
/* [uuid("ff1310fb-472e-5a85-965b-6fc0502e575f"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics2Vtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics2 * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics2 * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics2 * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics2 * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics2 * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics2 * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
HRESULT ( STDMETHODCALLTYPE *CreateFromShapeArrayAndDataArray )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics2 * This,
        /* [in] */unsigned int shapeLength,
        /* [size_is(shapeLength), in] */__int64 * shape,
        /* [in] */unsigned int dataLength,
        /* [size_is(dataLength), in] */boolean * data,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBoolean * * result
        );
    HRESULT ( STDMETHODCALLTYPE *CreateFromBuffer )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics2 * This,
        /* [in] */unsigned int shapeLength,
        /* [size_is(shapeLength), in] */__int64 * shape,
        /* [in] */__x_ABI_CWindows_CStorage_CStreams_CIBuffer * buffer,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBoolean * * result
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics2Vtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics2
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics2Vtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics2_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics2_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics2_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics2_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics2_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics2_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics2_CreateFromShapeArrayAndDataArray(This,shapeLength,shape,dataLength,data,result) \
    ( (This)->lpVtbl->CreateFromShapeArrayAndDataArray(This,shapeLength,shape,dataLength,data,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics2_CreateFromBuffer(This,shapeLength,shape,buffer,result) \
    ( (This)->lpVtbl->CreateFromBuffer(This,shapeLength,shape,buffer,result) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics2;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorBooleanStatics2_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000


/*
 *
 * Class Microsoft.AI.MachineLearning.TensorBoolean
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * RuntimeClass contains static methods.
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorBooleanStatics2 interface starting with version 2.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorBooleanStatics interface starting with version 1.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.ITensorBoolean ** Default Interface **
 *    Microsoft.AI.MachineLearning.ITensor
 *    Windows.Foundation.IMemoryBuffer
 *
 * Class Threading Model:  Both Single and Multi Threaded Apartment
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorBoolean_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorBoolean_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_TensorBoolean[] = L"Microsoft.AI.MachineLearning.TensorBoolean";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000



/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorDouble
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorDouble
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorDouble:HRESULT GetAsVectorView(Windows.Foundation.Collections.IVectorView`1<Double>**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDouble_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDouble_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorDouble[] = L"Microsoft.AI.MachineLearning.ITensorDouble";
/* [uuid("742a3b98-b0ea-59b3-b0f5-f257608df0c3"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDouble * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDouble * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDouble * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDouble * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDouble * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDouble * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
HRESULT ( STDMETHODCALLTYPE *GetAsVectorView )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDouble * This,
        /* [out, retval] */__FIVectorView_1_double * * result
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDouble
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDouble_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDouble_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDouble_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDouble_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDouble_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDouble_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDouble_GetAsVectorView(This,result) \
    ( (This)->lpVtbl->GetAsVectorView(This,result) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDouble;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDouble_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorDoubleStatics
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorDouble
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorDoubleStatics:HRESULT Create(Microsoft.AI.MachineLearning.TensorDouble**);HRESULT Create2(Windows.Foundation.Collections.IIterable`1<Int64>*,Microsoft.AI.MachineLearning.TensorDouble**);HRESULT CreateFromArray(Windows.Foundation.Collections.IIterable`1<Int64>*,UInt32,Double*,Microsoft.AI.MachineLearning.TensorDouble**);HRESULT CreateFromIterable(Windows.Foundation.Collections.IIterable`1<Int64>*,Windows.Foundation.Collections.IIterable`1<Double>*,Microsoft.AI.MachineLearning.TensorDouble**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorDoubleStatics[] = L"Microsoft.AI.MachineLearning.ITensorDoubleStatics";
/* [uuid("2c4821cf-0cb2-567b-b8d5-980bae99c2e8"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStaticsVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
/* [overload] */HRESULT ( STDMETHODCALLTYPE *Create )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics * This,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDouble * * result
        );
    /* [overload] */HRESULT ( STDMETHODCALLTYPE *Create2 )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics * This,
        /* [in] */__FIIterable_1___z__zint64 * shape,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDouble * * result
        );
    HRESULT ( STDMETHODCALLTYPE *CreateFromArray )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics * This,
        /* [in] */__FIIterable_1___z__zint64 * shape,
        /* [in] */unsigned int dataLength,
        /* [size_is(dataLength), in] */double * data,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDouble * * result
        );
    HRESULT ( STDMETHODCALLTYPE *CreateFromIterable )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics * This,
        /* [in] */__FIIterable_1___z__zint64 * shape,
        /* [in] */__FIIterable_1_double * data,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDouble * * result
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStaticsVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStaticsVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics_Create(This,result) \
    ( (This)->lpVtbl->Create(This,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics_Create2(This,shape,result) \
    ( (This)->lpVtbl->Create2(This,shape,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics_CreateFromArray(This,shape,dataLength,data,result) \
    ( (This)->lpVtbl->CreateFromArray(This,shape,dataLength,data,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics_CreateFromIterable(This,shape,data,result) \
    ( (This)->lpVtbl->CreateFromIterable(This,shape,data,result) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorDoubleStatics2
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 2.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorDouble
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorDoubleStatics2:HRESULT CreateFromShapeArrayAndDataArray(UInt32,Int64*,UInt32,Double*,Microsoft.AI.MachineLearning.TensorDouble**);HRESULT CreateFromBuffer(UInt32,Int64*,Windows.Storage.Streams.IBuffer*,Microsoft.AI.MachineLearning.TensorDouble**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics2_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics2_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorDoubleStatics2[] = L"Microsoft.AI.MachineLearning.ITensorDoubleStatics2";
/* [uuid("1a4f5b44-280a-5947-97c2-4fc2fcdbc65b"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics2Vtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics2 * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics2 * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics2 * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics2 * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics2 * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics2 * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
HRESULT ( STDMETHODCALLTYPE *CreateFromShapeArrayAndDataArray )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics2 * This,
        /* [in] */unsigned int shapeLength,
        /* [size_is(shapeLength), in] */__int64 * shape,
        /* [in] */unsigned int dataLength,
        /* [size_is(dataLength), in] */double * data,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDouble * * result
        );
    HRESULT ( STDMETHODCALLTYPE *CreateFromBuffer )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics2 * This,
        /* [in] */unsigned int shapeLength,
        /* [size_is(shapeLength), in] */__int64 * shape,
        /* [in] */__x_ABI_CWindows_CStorage_CStreams_CIBuffer * buffer,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDouble * * result
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics2Vtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics2
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics2Vtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics2_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics2_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics2_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics2_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics2_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics2_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics2_CreateFromShapeArrayAndDataArray(This,shapeLength,shape,dataLength,data,result) \
    ( (This)->lpVtbl->CreateFromShapeArrayAndDataArray(This,shapeLength,shape,dataLength,data,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics2_CreateFromBuffer(This,shapeLength,shape,buffer,result) \
    ( (This)->lpVtbl->CreateFromBuffer(This,shapeLength,shape,buffer,result) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics2;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorDoubleStatics2_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000


/*
 *
 * Class Microsoft.AI.MachineLearning.TensorDouble
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * RuntimeClass contains static methods.
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorDoubleStatics2 interface starting with version 2.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorDoubleStatics interface starting with version 1.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.ITensorDouble ** Default Interface **
 *    Microsoft.AI.MachineLearning.ITensor
 *    Windows.Foundation.IMemoryBuffer
 *
 * Class Threading Model:  Both Single and Multi Threaded Apartment
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorDouble_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorDouble_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_TensorDouble[] = L"Microsoft.AI.MachineLearning.TensorDouble";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000



/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorString
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorString
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorString:HRESULT GetAsVectorView(Windows.Foundation.Collections.IVectorView`1<String>**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorString_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorString_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorString[] = L"Microsoft.AI.MachineLearning.ITensorString";
/* [uuid("32e5741f-0457-5df4-aea4-812fc70b138f"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorString * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorString * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorString * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorString * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorString * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorString * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
HRESULT ( STDMETHODCALLTYPE *GetAsVectorView )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorString * This,
        /* [out, retval] */__FIVectorView_1_HSTRING * * result
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorString
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorString_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorString_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorString_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorString_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorString_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorString_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorString_GetAsVectorView(This,result) \
    ( (This)->lpVtbl->GetAsVectorView(This,result) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorString;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorString_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorStringStatics
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorString
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorStringStatics:HRESULT Create(Microsoft.AI.MachineLearning.TensorString**);HRESULT Create2(Windows.Foundation.Collections.IIterable`1<Int64>*,Microsoft.AI.MachineLearning.TensorString**);HRESULT CreateFromArray(Windows.Foundation.Collections.IIterable`1<Int64>*,UInt32,String*,Microsoft.AI.MachineLearning.TensorString**);HRESULT CreateFromIterable(Windows.Foundation.Collections.IIterable`1<Int64>*,Windows.Foundation.Collections.IIterable`1<String>*,Microsoft.AI.MachineLearning.TensorString**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorStringStatics[] = L"Microsoft.AI.MachineLearning.ITensorStringStatics";
/* [uuid("a27ef7bd-b6d5-5d3d-bbbf-9012d76733e5"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStaticsVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
/* [overload] */HRESULT ( STDMETHODCALLTYPE *Create )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics * This,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorString * * result
        );
    /* [overload] */HRESULT ( STDMETHODCALLTYPE *Create2 )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics * This,
        /* [in] */__FIIterable_1___z__zint64 * shape,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorString * * result
        );
    HRESULT ( STDMETHODCALLTYPE *CreateFromArray )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics * This,
        /* [in] */__FIIterable_1___z__zint64 * shape,
        /* [in] */unsigned int dataLength,
        /* [size_is(dataLength), in] */HSTRING * data,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorString * * result
        );
    HRESULT ( STDMETHODCALLTYPE *CreateFromIterable )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics * This,
        /* [in] */__FIIterable_1___z__zint64 * shape,
        /* [in] */__FIIterable_1_HSTRING * data,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorString * * result
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStaticsVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStaticsVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics_Create(This,result) \
    ( (This)->lpVtbl->Create(This,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics_Create2(This,shape,result) \
    ( (This)->lpVtbl->Create2(This,shape,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics_CreateFromArray(This,shape,dataLength,data,result) \
    ( (This)->lpVtbl->CreateFromArray(This,shape,dataLength,data,result) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics_CreateFromIterable(This,shape,data,result) \
    ( (This)->lpVtbl->CreateFromIterable(This,shape,data,result) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.ITensorStringStatics2
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 2.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.TensorString
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.ITensorStringStatics2:HRESULT CreateFromShapeArrayAndDataArray(UInt32,Int64*,UInt32,String*,Microsoft.AI.MachineLearning.TensorString**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics2_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics2_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_ITensorStringStatics2[] = L"Microsoft.AI.MachineLearning.ITensorStringStatics2";
/* [uuid("1984cf8a-b839-5bc8-b2b5-865f7867edd4"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics2Vtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics2 * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics2 * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics2 * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics2 * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics2 * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics2 * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
HRESULT ( STDMETHODCALLTYPE *CreateFromShapeArrayAndDataArray )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics2 * This,
        /* [in] */unsigned int shapeLength,
        /* [size_is(shapeLength), in] */__int64 * shape,
        /* [in] */unsigned int dataLength,
        /* [size_is(dataLength), in] */HSTRING * data,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorString * * result
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics2Vtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics2
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics2Vtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics2_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics2_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics2_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics2_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics2_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics2_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics2_CreateFromShapeArrayAndDataArray(This,shapeLength,shape,dataLength,data,result) \
    ( (This)->lpVtbl->CreateFromShapeArrayAndDataArray(This,shapeLength,shape,dataLength,data,result) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics2;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CITensorStringStatics2_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x20000


/*
 *
 * Class Microsoft.AI.MachineLearning.TensorString
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * RuntimeClass contains static methods.
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorStringStatics2 interface starting with version 2.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *   Static Methods exist on the Microsoft.AI.MachineLearning.ITensorStringStatics interface starting with version 1.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.ITensorString ** Default Interface **
 *    Microsoft.AI.MachineLearning.ITensor
 *    Windows.Foundation.IMemoryBuffer
 *
 * Class Threading Model:  Both Single and Multi Threaded Apartment
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorString_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_TensorString_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_TensorString[] = L"Microsoft.AI.MachineLearning.TensorString";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000



/*
 *
 * Interface Microsoft.AI.MachineLearning.IImageFeatureValue
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.ImageFeatureValue
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.IImageFeatureValue:HRESULT get_VideoFrame(Windows.Media.VideoFrame**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValue_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValue_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_IImageFeatureValue[] = L"Microsoft.AI.MachineLearning.IImageFeatureValue";
/* [uuid("ebec66ba-18d9-5851-8bed-86559cbc3545"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValueVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValue * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValue * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValue * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValue * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValue * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValue * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
/* [propget] */HRESULT ( STDMETHODCALLTYPE *get_VideoFrame )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValue * This,
        /* [out, retval] */__x_ABI_CWindows_CMedia_CIVideoFrame * * value
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValueVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValue
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValueVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValue_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValue_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValue_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValue_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValue_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValue_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValue_get_VideoFrame(This,value) \
    ( (This)->lpVtbl->get_VideoFrame(This,value) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValue;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValue_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Interface Microsoft.AI.MachineLearning.IImageFeatureValueStatics
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Interface is a part of the implementation of type Microsoft.AI.MachineLearning.ImageFeatureValue
 *
 *
 * The IID for this interface was automatically generated by MIDLRT.
 *
 * Interface IID generation seed: Microsoft.AI.MachineLearning.IImageFeatureValueStatics:HRESULT CreateFromVideoFrame(Windows.Media.VideoFrame*,Microsoft.AI.MachineLearning.ImageFeatureValue**);
 *
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000
#if !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValueStatics_INTERFACE_DEFINED__)
#define ____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValueStatics_INTERFACE_DEFINED__
extern const __declspec(selectany) _Null_terminated_ WCHAR InterfaceName_Microsoft_AI_MachineLearning_IImageFeatureValueStatics[] = L"Microsoft.AI.MachineLearning.IImageFeatureValueStatics";
/* [uuid("c822d410-8e0e-5fec-9c61-3cb55cdd8590"), contract, object, exclusiveto] */
typedef struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValueStaticsVtbl
{
    BEGIN_INTERFACE
    HRESULT ( STDMETHODCALLTYPE *QueryInterface)(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValueStatics * This,
    /* [in] */ __RPC__in REFIID riid,
    /* [annotation][iid_is][out] */
    _COM_Outptr_  void **ppvObject
    );

ULONG ( STDMETHODCALLTYPE *AddRef )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValueStatics * This
    );

ULONG ( STDMETHODCALLTYPE *Release )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValueStatics * This
    );

HRESULT ( STDMETHODCALLTYPE *GetIids )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValueStatics * This,
    /* [out] */ __RPC__out ULONG *iidCount,
    /* [size_is][size_is][out] */ __RPC__deref_out_ecount_full_opt(*iidCount) IID **iids
    );

HRESULT ( STDMETHODCALLTYPE *GetRuntimeClassName )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValueStatics * This,
    /* [out] */ __RPC__deref_out_opt HSTRING *className
    );

HRESULT ( STDMETHODCALLTYPE *GetTrustLevel )(
    __RPC__in __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValueStatics * This,
    /* [OUT ] */ __RPC__out TrustLevel *trustLevel
    );
HRESULT ( STDMETHODCALLTYPE *CreateFromVideoFrame )(
        __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValueStatics * This,
        /* [in] */__x_ABI_CWindows_CMedia_CIVideoFrame * image,
        /* [out, retval] */__x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValue * * result
        );
    END_INTERFACE
    
} __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValueStaticsVtbl;

interface __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValueStatics
{
    CONST_VTBL struct __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValueStaticsVtbl *lpVtbl;
};

#ifdef COBJMACROS
#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValueStatics_QueryInterface(This,riid,ppvObject) \
( (This)->lpVtbl->QueryInterface(This,riid,ppvObject) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValueStatics_AddRef(This) \
        ( (This)->lpVtbl->AddRef(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValueStatics_Release(This) \
        ( (This)->lpVtbl->Release(This) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValueStatics_GetIids(This,iidCount,iids) \
        ( (This)->lpVtbl->GetIids(This,iidCount,iids) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValueStatics_GetRuntimeClassName(This,className) \
        ( (This)->lpVtbl->GetRuntimeClassName(This,className) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValueStatics_GetTrustLevel(This,trustLevel) \
        ( (This)->lpVtbl->GetTrustLevel(This,trustLevel) )

#define __x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValueStatics_CreateFromVideoFrame(This,image,result) \
    ( (This)->lpVtbl->CreateFromVideoFrame(This,image,result) )


#endif /* COBJMACROS */


EXTERN_C const IID IID___x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValueStatics;
#endif /* !defined(____x_ABI_CMicrosoft_CAI_CMachineLearning_CIImageFeatureValueStatics_INTERFACE_DEFINED__) */
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


/*
 *
 * Class Microsoft.AI.MachineLearning.ImageFeatureValue
 *
 * Introduced to Microsoft.AI.MachineLearning.MachineLearningContract in version 1.0
 *
 *
 * Type can be called by both modern windows apps and classic windows apps
 *
 * RuntimeClass contains static methods.
 *   Static Methods exist on the Microsoft.AI.MachineLearning.IImageFeatureValueStatics interface starting with version 1.0 of the Microsoft.AI.MachineLearning.MachineLearningContract API contract
 *
 * Class implements the following interfaces:
 *    Microsoft.AI.MachineLearning.IImageFeatureValue ** Default Interface **
 *    Microsoft.AI.MachineLearning.ILearningModelFeatureValue
 *
 * Class Threading Model:  Both Single and Multi Threaded Apartment
 *
 * Class Marshaling Behavior:  Agile - Class is agile
 *
 */
#if MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000

#ifndef RUNTIMECLASS_Microsoft_AI_MachineLearning_ImageFeatureValue_DEFINED
#define RUNTIMECLASS_Microsoft_AI_MachineLearning_ImageFeatureValue_DEFINED
extern const __declspec(selectany) _Null_terminated_ WCHAR RuntimeClass_Microsoft_AI_MachineLearning_ImageFeatureValue[] = L"Microsoft.AI.MachineLearning.ImageFeatureValue";
#endif
#endif // MICROSOFT_AI_MACHINELEARNING_MACHINELEARNINGCONTRACT_VERSION >= 0x10000


#endif // defined(__cplusplus)
#pragma pop_macro("MIDL_CONST_ID")
#endif // __Microsoft2EAI2EMachineLearning_p_h__

#endif // __Microsoft2EAI2EMachineLearning_h__
