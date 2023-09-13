import subprocess
import sys
import os

target_zip = os.path.join(os.path.dirname(__file__), '_debugger_case_subprocess_target.zip')


def create_zip():
    import zipfile
    z = zipfile.ZipFile(target_zip, 'w', zipfile.ZIP_DEFLATED)
    z.writestr('__main__.py', '''
import _debugger_case_subprocess_zip
_debugger_case_subprocess_zip.call_from_zip()
''')


def call_from_zip():
    print('Called from zip')  # break here


def main():
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.dirname(__file__)
    result = subprocess.check_output([sys.executable, target_zip], env=env).decode('utf-8')
    assert 'Called from zip' in result
    print('TEST SUCEEDED!')


if __name__ == '__main__':
    if not os.path.exists(target_zip):
        create_zip()
    assert os.path.exists(target_zip)

    main()
