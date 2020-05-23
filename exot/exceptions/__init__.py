"""Exception classes

Exceptions are put in a single module primarily to allow using exceptions without the
risk of cyclic dependencies between modules, and a more legible exception path.
"""

from ._factory import ErrorFactory as __

# fmt: off

"""
Utilities errors
================
"""

# Generic factory errors
GenericFactoryErrorFactory = __("GenericFactory")
GenericFactoryException = GenericFactoryErrorFactory()
AmbiguousMatchError = GenericFactoryErrorFactory("AmbiguousMatch", ValueError)
AbstractMatchError = GenericFactoryErrorFactory("AbstractMatch", TypeError)
NothingMatchedError = GenericFactoryErrorFactory("NothingMatched", ValueError)

# File manipulation errors
WrongRootPathError = __()("WrongRootPath", RuntimeError)

# Other
EvalTypeError = __()("BadResultingType", TypeError)
MatchNotFound = __()("MatchNotFound", ValueError, RuntimeError)

"""
Mixins errors
-------------
"""

# Exceptions for Configurable
ConfigurableErrorFactory = __("Configurable")
ConfigurableException = ConfigurableErrorFactory()
RequiredKeysMissingError = ConfigurableErrorFactory("RequiredKeysMissing", ValueError)
ConfigError = ConfigurableErrorFactory("ConfigError", ValueError)
ConfigMissingError = ConfigurableErrorFactory("ConfigMissing", ValueError)
MisconfiguredError = ConfigurableErrorFactory("Misconfigured", RuntimeError)

# Exceptions for HasParent
HasParentErrorFactory = __("HasParent")
HasParentException = HasParentErrorFactory()
HasParentTypeError = HasParentErrorFactory("TypeError", TypeError)
WrongParentTypeError = HasParentErrorFactory("WrongParentType", TypeError)

# Exceptions for SubclassTracker
SubclassTrackerErrorFactory = __("SubclassTracker")
SubclassTrackerException = SubclassTrackerErrorFactory()
SubclassTrackerValueError = SubclassTrackerErrorFactory.common[ValueError]
SubclassTrackerTypeError = SubclassTrackerErrorFactory.common[TypeError]

# Exceptions for Serialisable
SerialisableErrorFactory = __("Serialisable")
SerialisableException = SerialisableErrorFactory()
SerialisableValueError = SerialisableErrorFactory.common[ValueError]
SerialisableTypeError = SerialisableErrorFactory.common[TypeError]
SerialisableDataLoadError = SerialisableErrorFactory("DataLoadError", ValueError, RuntimeError)
SerialisableDataSaveError = SerialisableErrorFactory("DataSaveError", ValueError, RuntimeError)

# Exceptions for Serialisable
IntermediatesErrorFactory = __("Intermediates")
IntermediatesException = IntermediatesErrorFactory()
IntermediatesValueError = IntermediatesErrorFactory.common[ValueError]
IntermediatesTypeError = IntermediatesErrorFactory.common[TypeError]
IntermediatesMissing = IntermediatesErrorFactory("IntermediatesMissing", RuntimeError)

"""
Driver errors
-------------
"""

# Backend errors
BackendErrorFactory = __("Backend")
BackendException = BackendErrorFactory()
BackendTypeError = BackendErrorFactory.common[TypeError]
BackendValueError = BackendErrorFactory.common[ValueError]
CouldNotConnectError = BackendErrorFactory("CouldNotConnect", ConnectionError, MisconfiguredError)
NotConnectedError = BackendErrorFactory("NotConnected", ConnectionError)

# Remote files
RemoteFileNotFoundError = BackendErrorFactory("RemoteFileNotFound", RuntimeError, FileNotFoundError)
RemoteCommandFailedError = BackendErrorFactory("RemoteCommandFailed", RuntimeError)
RemoteSetupError = BackendErrorFactory("RemoteSetupError", RuntimeError)
TransferFailedError = BackendErrorFactory("TransferFailed", ConnectionError, RuntimeError)

# Driver errors
DriverErrorFactory = __("Driver")
DriverException = DriverErrorFactory()
FileOperationFailedError = DriverErrorFactory()
DriverValueError = DriverErrorFactory.common[ValueError]
DriverTypeError = DriverErrorFactory.common[TypeError]
PlatformLocked = DriverErrorFactory("PlatformLocked", RuntimeError)

"""
Experiment errors
=================
"""

# Experiment errors
ExperimentErrorFactory = __("Experiment")
ExperimentException = ExperimentErrorFactory()
ExperimentValueError = ExperimentErrorFactory.common[ValueError]
ExperimentTypeError = ExperimentErrorFactory.common[TypeError]
ExperimentRuntimeError = ExperimentErrorFactory.common[RuntimeError]
WrongDriverError = ExperimentErrorFactory("WrongDriver", ValueError)
DriverCreationError = ExperimentErrorFactory("DriverCreationFailed", RuntimeError)
RequiredLayerMissingError = ExperimentErrorFactory("RequiredLayerMissing", ValueError)
SerialisingAbstractError = ExperimentErrorFactory("SerialisingAbstract", ValueError)
ExperimentAbortedError = ExperimentErrorFactory("ExperimentAborted", RuntimeError)
InvalidBackupPathError = ExperimentErrorFactory("InvalidBackupPath", OSError, RuntimeError)
ExperimentExecutionFailed = ExperimentErrorFactory("ExperimentExecutionFailed", RuntimeError)

GenerateValueAssertion = ExperimentErrorFactory("GenerateValueAssertion", AssertionError, ValueError)
GenerateTypeAssertion = ExperimentErrorFactory("GenerateTypeAssertion", AssertionError, TypeError)

"""
Layer errors
------------
"""

LayerErrorFactory = __("Layer")
LayerTypeError = LayerErrorFactory.common[TypeError]
LayerValueError = LayerErrorFactory.common[ValueError]
TypeValidationFailed = LayerErrorFactory("TypeValidationFailed", TypeError)
ValueValidationFailed = LayerErrorFactory("ValueValidationFailed", ValueError)
LayerConfigMissing = LayerErrorFactory("LayerConfigMissing", ConfigMissingError)
LayerMisconfigured = LayerErrorFactory("LayerMisconfigured", ValueError, TypeError)

LayerRuntimeError = LayerErrorFactory.common[RuntimeError]
EnvironmentMissing = LayerErrorFactory("EnvironmentMissing", ValueError, RuntimeError)
LayerRuntimeMisconfigured = LayerErrorFactory("LayerRuntimeMisconfigured", ValueError, RuntimeError)
