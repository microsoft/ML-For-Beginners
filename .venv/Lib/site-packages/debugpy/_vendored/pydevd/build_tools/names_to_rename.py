'''
Helper module to hold the names to rename while doing refactoring to convert to pep8.
'''
NAMES = '''
# sendCaughtExceptionStack
# sendBreakpointConditionException
# setSuspend
# processThreadNotAlive
# sendCaughtExceptionStackProceeded
# doWaitSuspend
# SetTraceForFrameAndParents
# prepareToRun
# processCommandLine
# initStdoutRedirect
# initStderrRedirect
# OnRun
# doKillPydevThread
# stopTrace
# handleExcept
# processCommand
# processNetCommand
# addCommand
# StartClient
# getNextSeq
# makeMessage
# StartServer
# threadToXML
# makeErrorMessage
# makeThreadCreatedMessage
# makeCustomFrameCreatedMessage
# makeListThreadsMessage
# makeVariableChangedMessage
# makeIoMessage
# makeVersionMessage
# makeThreadKilledMessage
# makeThreadSuspendStr
# makeValidXmlValue
# makeThreadSuspendMessage
# makeThreadRunMessage
# makeGetVariableMessage
# makeGetArrayMessage
# makeGetFrameMessage
# makeEvaluateExpressionMessage
# makeGetCompletionsMessage
# makeGetFileContents
# makeSendBreakpointExceptionMessage
# makeSendCurrExceptionTraceMessage
# makeSendCurrExceptionTraceProceededMessage
# makeSendConsoleMessage
# makeCustomOperationMessage
# makeLoadSourceMessage
# makeShowConsoleMessage
# makeExitMessage
# canBeExecutedBy
# doIt
# additionalInfo
# cmdFactory
# GetExceptionTracebackStr
# _GetStackStr
# _InternalSetTrace
# ReplaceSysSetTraceFunc
# RestoreSysSetTraceFunc



# AddContent
# AddException
# AddObserver
# # Call -- skip
# # Call1 -- skip
# # Call2 -- skip
# # Call3 -- skip
# # Call4 -- skip
# ChangePythonPath
# CheckArgs
# CheckChar
# CompleteFromDir
# CreateDbFrame
# CustomFramesContainerInit
# DictContains
# DictItems
# DictIterItems
# DictIterValues
# DictKeys
# DictPop
# DictValues


# DoExit
# DoFind
# EndRedirect
# # Exec -- skip
# ExecuteTestsInParallel
# # Find -- skip
# FinishDebuggingSession
# FlattenTestSuite
# GenerateCompletionsAsXML
# GenerateImportsTipForModule
# GenerateTip


# testAddExec
# testComplete
# testCompleteDoesNotDoPythonMatches
# testCompletionSocketsAndMessages
# testConsoleHello
# testConsoleRequests
# testDotNetLibraries
# testEdit
# testGetCompletions
# testGetNamespace
# testGetReferrers1
# testGetReferrers2
# testGetReferrers3
# testGetReferrers4
# testGetReferrers5
# testGetReferrers6
# testGetReferrers7
# testGettingInfoOnJython
# testGui
# testHistory
# testImports
# testImports1
# testImports1a
# testImports1b
# testImports1c
# testImports2
# testImports2a
# testImports2b
# testImports2c
# testImports3
# testImports4
# testImports5
# testInspect
# testIt
# testMessage
# testPrint
# testProperty
# testProperty2
# testProperty3
# testQuestionMark
# testSearch
# testSearchOnJython
# testServer
# testTipOnString
# toXML
# updateCustomFrame
# varToXML

#
# GetContents
# GetCoverageFiles
# GetFile
# GetFileNameAndBaseFromFile
# GetFilenameAndBase
# GetFrame
# GetGlobalDebugger # -- renamed but kept backward-compatibility
# GetNormPathsAndBase
# GetNormPathsAndBaseFromFile
# GetTestsToRun -- skip
# GetThreadId
# GetVmType
# IPythonEditor -- skip
# ImportName
# InitializeServer
# IterFrames


# Method1 -- skip
# Method1a  -- skip
# Method2 -- skip
# Method3 -- skip

# NewConsolidate
# NormFileToClient
# NormFileToServer
# # Notify -- skip
# # NotifyFinished -- skip
# OnFunButton
# # OnInit -- skip
# OnTimeToClose
# PydevdFindThreadById
# PydevdLog
# # RequestInput -- skip


# Search -- manual: search_definition
# ServerProxy -- skip
# SetGlobalDebugger

# SetServer
# SetUp
# SetTrace -- skip


# SetVmType
# SetupType
# StartCoverageSupport
# StartCoverageSupportFromParams
# StartPydevNosePluginSingleton
# StartRedirect
# ToTuple

# addAdditionalFrameById
# removeAdditionalFrameById
# removeCustomFrame
# addCustomFrame
# addError -- skip
# addExec
# addFailure -- skip
# addSuccess -- skip
# assertArgs
# assertIn

# basicAsStr
# changeAttrExpression
# # changeVariable -- skip (part of public API for console)
# checkOutput
# checkOutputRedirect
# clearBuffer

# # connectToDebugger -- skip (part of public API for console)
# connectToServer
# consoleExec
# createConnections
# createStdIn
# customOperation
# dirObj
# doAddExec
# doExecCode
# dumpFrames

# # enableGui -- skip (part of public API for console)
# evalInContext
# evaluateExpression
# # execLine  -- skip (part of public API for console)
# # execMultipleLines -- skip (part of public API for console)
# findFrame
# orig_findFrame
# finishExec
# fixGetpass

# forceServerKill
# formatArg
# formatCompletionMessage
# formatParamClassName
# frameVarsToXML
# fullyNormalizePath

# getArray -- skip (part of public API for console)
# getAsDoc
# getCapturedOutput
# getCompletions -- skip (part of public API for console)

# getCompletionsMessage
# getCustomFrame
# # getDescription -- skip (part of public API for console)
# getDictionary
# # getFrame -- skip (part of public API for console)
# getFrameName



# getFrameStack
# getFreeAddresses
# getInternalQueue
# getIoFromError
# getNamespace
# getTestName
# getTokenAndData
# getType

# getVariable -- skip (part of public API for console)

# # haveAliveThreads -> has_threads_alive
# initializeNetwork
# isThreadAlive
# # iterFrames -> _iter_frames
# # keyStr -> key_to_str
# killAllPydevThreads
# longRunning
# # metA -- skip
# nativePath

# needMore
# needMoreForCode
# # notifyCommands -- skip (part of public API)
# # notifyConnected -- skip (part of public API)
# # notifyStartTest -- skip (part of public API)
# # notifyTest -- skip (part of public API)
# # notifyTestRunFinished -- skip (part of public API)
# # notifyTestsCollected -- skip (part of public API)
# postInternalCommand
# processInternalCommands
# readMsg


# redirectStdout
# removeInvalidChars
# reportCond
# resolveCompoundVariable
# resolveVar
# restoreStdout
# sendKillMsg
# sendSignatureCallTrace
# setTracingForUntracedContexts
# startClientThread
# startDebuggerServerThread
# startExec

# startTest -- skip
# stopTest -- skip
# setUp -- skip
# setUpClass -- skip
# setUpModule -- skip
# tearDown -- skip

'''