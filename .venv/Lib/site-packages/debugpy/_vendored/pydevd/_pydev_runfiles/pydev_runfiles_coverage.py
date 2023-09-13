import os.path
import sys
from _pydevd_bundle.pydevd_constants import Null


#=======================================================================================================================
# get_coverage_files
#=======================================================================================================================
def get_coverage_files(coverage_output_dir, number_of_files):
    base_dir = coverage_output_dir
    ret = []
    i = 0
    while len(ret) < number_of_files:
        while True:
            f = os.path.join(base_dir, '.coverage.%s' % i)
            i += 1
            if not os.path.exists(f):
                ret.append(f)
                break #Break only inner for.
    return ret


#=======================================================================================================================
# start_coverage_support
#=======================================================================================================================
def start_coverage_support(configuration):
    return start_coverage_support_from_params(
        configuration.coverage_output_dir, 
        configuration.coverage_output_file, 
        configuration.jobs, 
        configuration.coverage_include, 
    )
    

#=======================================================================================================================
# start_coverage_support_from_params
#=======================================================================================================================
def start_coverage_support_from_params(coverage_output_dir, coverage_output_file, jobs, coverage_include):
    coverage_files = []
    coverage_instance = Null()
    if coverage_output_dir or coverage_output_file:
        try:
            import coverage #@UnresolvedImport
        except:
            sys.stderr.write('Error: coverage module could not be imported\n')
            sys.stderr.write('Please make sure that the coverage module (http://nedbatchelder.com/code/coverage/)\n')
            sys.stderr.write('is properly installed in your interpreter: %s\n' % (sys.executable,))
            
            import traceback;traceback.print_exc()
        else:
            if coverage_output_dir:
                if not os.path.exists(coverage_output_dir):
                    sys.stderr.write('Error: directory for coverage output (%s) does not exist.\n' % (coverage_output_dir,))
                    
                elif not os.path.isdir(coverage_output_dir):
                    sys.stderr.write('Error: expected (%s) to be a directory.\n' % (coverage_output_dir,))
                    
                else:
                    n = jobs
                    if n <= 0:
                        n += 1
                    n += 1 #Add 1 more for the current process (which will do the initial import).
                    coverage_files = get_coverage_files(coverage_output_dir, n)
                    os.environ['COVERAGE_FILE'] = coverage_files.pop(0)
                    
                    coverage_instance = coverage.coverage(source=[coverage_include])
                    coverage_instance.start()
                    
            elif coverage_output_file:
                #Client of parallel run.
                os.environ['COVERAGE_FILE'] = coverage_output_file
                coverage_instance = coverage.coverage(source=[coverage_include])
                coverage_instance.start()
                
    return coverage_files, coverage_instance

