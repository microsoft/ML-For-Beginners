import sys


def __getfilesystemencoding():
    '''
    Note: there's a copy of this method in interpreterInfo.py
    '''
    try:
        ret = sys.getfilesystemencoding()
        if not ret:
            raise RuntimeError('Unable to get encoding.')
        return ret
    except:
        try:
            #Handle Jython
            from java.lang import System  # @UnresolvedImport
            env = System.getProperty("os.name").lower()
            if env.find('win') != -1:
                return 'ISO-8859-1'  #mbcs does not work on Jython, so, use a (hopefully) suitable replacement
            return 'utf-8'
        except:
            pass

        #Only available from 2.3 onwards.
        if sys.platform == 'win32':
            return 'mbcs'
        return 'utf-8'

def getfilesystemencoding():
    try:
        ret = __getfilesystemencoding()

        #Check if the encoding is actually there to be used!
        if hasattr('', 'encode'):
            ''.encode(ret)
        if hasattr('', 'decode'):
            ''.decode(ret)

        return ret
    except:
        return 'utf-8'
