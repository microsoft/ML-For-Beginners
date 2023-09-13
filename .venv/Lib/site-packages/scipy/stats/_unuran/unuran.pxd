# File automatically generated using autopxd2

from libc.stdio cimport FILE

cdef extern from "unuran.h" nogil:

    cdef struct unur_distr

    ctypedef unur_distr UNUR_DISTR

    cdef struct unur_par

    ctypedef unur_par UNUR_PAR

    cdef struct unur_gen

    ctypedef unur_gen UNUR_GEN

    cdef struct unur_urng

    ctypedef unur_urng UNUR_URNG

    ctypedef double UNUR_FUNCT_CONT(double x, unur_distr* distr)

    ctypedef double UNUR_FUNCT_DISCR(int x, unur_distr* distr)

    ctypedef int UNUR_IFUNCT_DISCR(double x, unur_distr* distr)

    ctypedef double UNUR_FUNCT_CVEC(double* x, unur_distr* distr)

    ctypedef int UNUR_VFUNCT_CVEC(double* result, double* x, unur_distr* distr)

    ctypedef double UNUR_FUNCTD_CVEC(double* x, int coord, unur_distr* distr)

    cdef struct unur_slist

    ctypedef void UNUR_ERROR_HANDLER(char* objid, char* file, int line, char* errortype, int unur_errno, char* reason)

    UNUR_URNG* unur_get_default_urng()

    UNUR_URNG* unur_set_default_urng(UNUR_URNG* urng_new)

    UNUR_URNG* unur_set_default_urng_aux(UNUR_URNG* urng_new)

    UNUR_URNG* unur_get_default_urng_aux()

    int unur_set_urng(UNUR_PAR* parameters, UNUR_URNG* urng)

    UNUR_URNG* unur_chg_urng(UNUR_GEN* generator, UNUR_URNG* urng)

    UNUR_URNG* unur_get_urng(UNUR_GEN* generator)

    int unur_set_urng_aux(UNUR_PAR* parameters, UNUR_URNG* urng_aux)

    int unur_use_urng_aux_default(UNUR_PAR* parameters)

    int unur_chgto_urng_aux_default(UNUR_GEN* generator)

    UNUR_URNG* unur_chg_urng_aux(UNUR_GEN* generator, UNUR_URNG* urng_aux)

    UNUR_URNG* unur_get_urng_aux(UNUR_GEN* generator)

    double unur_urng_sample(UNUR_URNG* urng)

    double unur_sample_urng(UNUR_GEN* gen)

    int unur_urng_sample_array(UNUR_URNG* urng, double* X, int dim)

    int unur_urng_reset(UNUR_URNG* urng)

    int unur_urng_sync(UNUR_URNG* urng)

    int unur_urng_seed(UNUR_URNG* urng, unsigned long seed)

    int unur_urng_anti(UNUR_URNG* urng, int anti)

    int unur_urng_nextsub(UNUR_URNG* urng)

    int unur_urng_resetsub(UNUR_URNG* urng)

    int unur_gen_sync(UNUR_GEN* generator)

    int unur_gen_seed(UNUR_GEN* generator, unsigned long seed)

    int unur_gen_anti(UNUR_GEN* generator, int anti)

    int unur_gen_reset(UNUR_GEN* generator)

    int unur_gen_nextsub(UNUR_GEN* generator)

    int unur_gen_resetsub(UNUR_GEN* generator)

    ctypedef double (*_unur_urng_new_sampleunif_ft)(void* state)

    UNUR_URNG* unur_urng_new(_unur_urng_new_sampleunif_ft sampleunif, void* state)

    void unur_urng_free(UNUR_URNG* urng)

    ctypedef unsigned int (*_unur_urng_set_sample_array_samplearray_ft)(void* state, double* X, int dim)

    int unur_urng_set_sample_array(UNUR_URNG* urng, _unur_urng_set_sample_array_samplearray_ft samplearray)

    ctypedef void (*_unur_urng_set_sync_sync_ft)(void* state)

    int unur_urng_set_sync(UNUR_URNG* urng, _unur_urng_set_sync_sync_ft sync)

    ctypedef void (*_unur_urng_set_seed_setseed_ft)(void* state, unsigned long seed)

    int unur_urng_set_seed(UNUR_URNG* urng, _unur_urng_set_seed_setseed_ft setseed)

    ctypedef void (*_unur_urng_set_anti_setanti_ft)(void* state, int anti)

    int unur_urng_set_anti(UNUR_URNG* urng, _unur_urng_set_anti_setanti_ft setanti)

    ctypedef void (*_unur_urng_set_reset_reset_ft)(void* state)

    int unur_urng_set_reset(UNUR_URNG* urng, _unur_urng_set_reset_reset_ft reset)

    ctypedef void (*_unur_urng_set_nextsub_nextsub_ft)(void* state)

    int unur_urng_set_nextsub(UNUR_URNG* urng, _unur_urng_set_nextsub_nextsub_ft nextsub)

    ctypedef void (*_unur_urng_set_resetsub_resetsub_ft)(void* state)

    int unur_urng_set_resetsub(UNUR_URNG* urng, _unur_urng_set_resetsub_resetsub_ft resetsub)

    ctypedef void (*_unur_urng_set_delete_fpdelete_ft)(void* state)

    int unur_urng_set_delete(UNUR_URNG* urng, _unur_urng_set_delete_fpdelete_ft fpdelete)

    cdef enum:
        UNUR_DISTR_CONT
        UNUR_DISTR_CEMP
        UNUR_DISTR_CVEC
        UNUR_DISTR_CVEMP
        UNUR_DISTR_MATR
        UNUR_DISTR_DISCR

    void unur_distr_free(UNUR_DISTR* distribution)

    int unur_distr_set_name(UNUR_DISTR* distribution, char* name)

    char* unur_distr_get_name(UNUR_DISTR* distribution)

    int unur_distr_get_dim(UNUR_DISTR* distribution)

    unsigned int unur_distr_get_type(UNUR_DISTR* distribution)

    int unur_distr_is_cont(UNUR_DISTR* distribution)

    int unur_distr_is_cvec(UNUR_DISTR* distribution)

    int unur_distr_is_cemp(UNUR_DISTR* distribution)

    int unur_distr_is_cvemp(UNUR_DISTR* distribution)

    int unur_distr_is_discr(UNUR_DISTR* distribution)

    int unur_distr_is_matr(UNUR_DISTR* distribution)

    int unur_distr_set_extobj(UNUR_DISTR* distribution, void* extobj)

    void* unur_distr_get_extobj(UNUR_DISTR* distribution)

    UNUR_DISTR* unur_distr_clone(UNUR_DISTR* distr)

    UNUR_DISTR* unur_distr_cemp_new()

    int unur_distr_cemp_set_data(UNUR_DISTR* distribution, double* sample, int n_sample)

    int unur_distr_cemp_read_data(UNUR_DISTR* distribution, char* filename)

    int unur_distr_cemp_get_data(UNUR_DISTR* distribution, double** sample)

    int unur_distr_cemp_set_hist(UNUR_DISTR* distribution, double* prob, int n_prob, double xmin, double xmax)

    int unur_distr_cemp_set_hist_prob(UNUR_DISTR* distribution, double* prob, int n_prob)

    int unur_distr_cemp_set_hist_domain(UNUR_DISTR* distribution, double xmin, double xmax)

    int unur_distr_cemp_set_hist_bins(UNUR_DISTR* distribution, double* bins, int n_bins)

    UNUR_DISTR* unur_distr_cont_new()

    int unur_distr_cont_set_pdf(UNUR_DISTR* distribution, UNUR_FUNCT_CONT* pdf)

    int unur_distr_cont_set_dpdf(UNUR_DISTR* distribution, UNUR_FUNCT_CONT* dpdf)

    int unur_distr_cont_set_cdf(UNUR_DISTR* distribution, UNUR_FUNCT_CONT* cdf)

    int unur_distr_cont_set_invcdf(UNUR_DISTR* distribution, UNUR_FUNCT_CONT* invcdf)

    UNUR_FUNCT_CONT* unur_distr_cont_get_pdf(UNUR_DISTR* distribution)

    UNUR_FUNCT_CONT* unur_distr_cont_get_dpdf(UNUR_DISTR* distribution)

    UNUR_FUNCT_CONT* unur_distr_cont_get_cdf(UNUR_DISTR* distribution)

    UNUR_FUNCT_CONT* unur_distr_cont_get_invcdf(UNUR_DISTR* distribution)

    double unur_distr_cont_eval_pdf(double x, UNUR_DISTR* distribution)

    double unur_distr_cont_eval_dpdf(double x, UNUR_DISTR* distribution)

    double unur_distr_cont_eval_cdf(double x, UNUR_DISTR* distribution)

    double unur_distr_cont_eval_invcdf(double u, UNUR_DISTR* distribution)

    int unur_distr_cont_set_logpdf(UNUR_DISTR* distribution, UNUR_FUNCT_CONT* logpdf)

    int unur_distr_cont_set_dlogpdf(UNUR_DISTR* distribution, UNUR_FUNCT_CONT* dlogpdf)

    int unur_distr_cont_set_logcdf(UNUR_DISTR* distribution, UNUR_FUNCT_CONT* logcdf)

    UNUR_FUNCT_CONT* unur_distr_cont_get_logpdf(UNUR_DISTR* distribution)

    UNUR_FUNCT_CONT* unur_distr_cont_get_dlogpdf(UNUR_DISTR* distribution)

    UNUR_FUNCT_CONT* unur_distr_cont_get_logcdf(UNUR_DISTR* distribution)

    double unur_distr_cont_eval_logpdf(double x, UNUR_DISTR* distribution)

    double unur_distr_cont_eval_dlogpdf(double x, UNUR_DISTR* distribution)

    double unur_distr_cont_eval_logcdf(double x, UNUR_DISTR* distribution)

    int unur_distr_cont_set_pdfstr(UNUR_DISTR* distribution, char* pdfstr)

    int unur_distr_cont_set_cdfstr(UNUR_DISTR* distribution, char* cdfstr)

    char* unur_distr_cont_get_pdfstr(UNUR_DISTR* distribution)

    char* unur_distr_cont_get_dpdfstr(UNUR_DISTR* distribution)

    char* unur_distr_cont_get_cdfstr(UNUR_DISTR* distribution)

    int unur_distr_cont_set_pdfparams(UNUR_DISTR* distribution, double* params, int n_params)

    int unur_distr_cont_get_pdfparams(UNUR_DISTR* distribution, double** params)

    int unur_distr_cont_set_pdfparams_vec(UNUR_DISTR* distribution, int par, double* param_vec, int n_param_vec)

    int unur_distr_cont_get_pdfparams_vec(UNUR_DISTR* distribution, int par, double** param_vecs)

    int unur_distr_cont_set_logpdfstr(UNUR_DISTR* distribution, char* logpdfstr)

    char* unur_distr_cont_get_logpdfstr(UNUR_DISTR* distribution)

    char* unur_distr_cont_get_dlogpdfstr(UNUR_DISTR* distribution)

    int unur_distr_cont_set_logcdfstr(UNUR_DISTR* distribution, char* logcdfstr)

    char* unur_distr_cont_get_logcdfstr(UNUR_DISTR* distribution)

    int unur_distr_cont_set_domain(UNUR_DISTR* distribution, double left, double right)

    int unur_distr_cont_get_domain(UNUR_DISTR* distribution, double* left, double* right)

    int unur_distr_cont_get_truncated(UNUR_DISTR* distribution, double* left, double* right)

    int unur_distr_cont_set_hr(UNUR_DISTR* distribution, UNUR_FUNCT_CONT* hazard)

    UNUR_FUNCT_CONT* unur_distr_cont_get_hr(UNUR_DISTR* distribution)

    double unur_distr_cont_eval_hr(double x, UNUR_DISTR* distribution)

    int unur_distr_cont_set_hrstr(UNUR_DISTR* distribution, char* hrstr)

    char* unur_distr_cont_get_hrstr(UNUR_DISTR* distribution)

    int unur_distr_cont_set_mode(UNUR_DISTR* distribution, double mode)

    int unur_distr_cont_upd_mode(UNUR_DISTR* distribution)

    double unur_distr_cont_get_mode(UNUR_DISTR* distribution)

    int unur_distr_cont_set_center(UNUR_DISTR* distribution, double center)

    double unur_distr_cont_get_center(UNUR_DISTR* distribution)

    int unur_distr_cont_set_pdfarea(UNUR_DISTR* distribution, double area)

    int unur_distr_cont_upd_pdfarea(UNUR_DISTR* distribution)

    double unur_distr_cont_get_pdfarea(UNUR_DISTR* distribution)

    UNUR_DISTR* unur_distr_cxtrans_new(UNUR_DISTR* distribution)

    UNUR_DISTR* unur_distr_cxtrans_get_distribution(UNUR_DISTR* distribution)

    int unur_distr_cxtrans_set_alpha(UNUR_DISTR* distribution, double alpha)

    int unur_distr_cxtrans_set_rescale(UNUR_DISTR* distribution, double mu, double sigma)

    double unur_distr_cxtrans_get_alpha(UNUR_DISTR* distribution)

    double unur_distr_cxtrans_get_mu(UNUR_DISTR* distribution)

    double unur_distr_cxtrans_get_sigma(UNUR_DISTR* distribution)

    int unur_distr_cxtrans_set_logpdfpole(UNUR_DISTR* distribution, double logpdfpole, double dlogpdfpole)

    int unur_distr_cxtrans_set_domain(UNUR_DISTR* distribution, double left, double right)

    UNUR_DISTR* unur_distr_corder_new(UNUR_DISTR* distribution, int n, int k)

    UNUR_DISTR* unur_distr_corder_get_distribution(UNUR_DISTR* distribution)

    int unur_distr_corder_set_rank(UNUR_DISTR* distribution, int n, int k)

    int unur_distr_corder_get_rank(UNUR_DISTR* distribution, int* n, int* k)

    UNUR_DISTR* unur_distr_cvec_new(int dim)

    int unur_distr_cvec_set_pdf(UNUR_DISTR* distribution, UNUR_FUNCT_CVEC* pdf)

    int unur_distr_cvec_set_dpdf(UNUR_DISTR* distribution, UNUR_VFUNCT_CVEC* dpdf)

    int unur_distr_cvec_set_pdpdf(UNUR_DISTR* distribution, UNUR_FUNCTD_CVEC* pdpdf)

    UNUR_FUNCT_CVEC* unur_distr_cvec_get_pdf(UNUR_DISTR* distribution)

    UNUR_VFUNCT_CVEC* unur_distr_cvec_get_dpdf(UNUR_DISTR* distribution)

    UNUR_FUNCTD_CVEC* unur_distr_cvec_get_pdpdf(UNUR_DISTR* distribution)

    double unur_distr_cvec_eval_pdf(double* x, UNUR_DISTR* distribution)

    int unur_distr_cvec_eval_dpdf(double* result, double* x, UNUR_DISTR* distribution)

    double unur_distr_cvec_eval_pdpdf(double* x, int coord, UNUR_DISTR* distribution)

    int unur_distr_cvec_set_logpdf(UNUR_DISTR* distribution, UNUR_FUNCT_CVEC* logpdf)

    int unur_distr_cvec_set_dlogpdf(UNUR_DISTR* distribution, UNUR_VFUNCT_CVEC* dlogpdf)

    int unur_distr_cvec_set_pdlogpdf(UNUR_DISTR* distribution, UNUR_FUNCTD_CVEC* pdlogpdf)

    UNUR_FUNCT_CVEC* unur_distr_cvec_get_logpdf(UNUR_DISTR* distribution)

    UNUR_VFUNCT_CVEC* unur_distr_cvec_get_dlogpdf(UNUR_DISTR* distribution)

    UNUR_FUNCTD_CVEC* unur_distr_cvec_get_pdlogpdf(UNUR_DISTR* distribution)

    double unur_distr_cvec_eval_logpdf(double* x, UNUR_DISTR* distribution)

    int unur_distr_cvec_eval_dlogpdf(double* result, double* x, UNUR_DISTR* distribution)

    double unur_distr_cvec_eval_pdlogpdf(double* x, int coord, UNUR_DISTR* distribution)

    int unur_distr_cvec_set_mean(UNUR_DISTR* distribution, double* mean)

    double* unur_distr_cvec_get_mean(UNUR_DISTR* distribution)

    int unur_distr_cvec_set_covar(UNUR_DISTR* distribution, double* covar)

    int unur_distr_cvec_set_covar_inv(UNUR_DISTR* distribution, double* covar_inv)

    double* unur_distr_cvec_get_covar(UNUR_DISTR* distribution)

    double* unur_distr_cvec_get_cholesky(UNUR_DISTR* distribution)

    double* unur_distr_cvec_get_covar_inv(UNUR_DISTR* distribution)

    int unur_distr_cvec_set_rankcorr(UNUR_DISTR* distribution, double* rankcorr)

    double* unur_distr_cvec_get_rankcorr(UNUR_DISTR* distribution)

    double* unur_distr_cvec_get_rk_cholesky(UNUR_DISTR* distribution)

    int unur_distr_cvec_set_marginals(UNUR_DISTR* distribution, UNUR_DISTR* marginal)

    int unur_distr_cvec_set_marginal_array(UNUR_DISTR* distribution, UNUR_DISTR** marginals)

    int unur_distr_cvec_set_marginal_list(UNUR_DISTR* distribution)

    UNUR_DISTR* unur_distr_cvec_get_marginal(UNUR_DISTR* distribution, int n)

    int unur_distr_cvec_set_pdfparams(UNUR_DISTR* distribution, double* params, int n_params)

    int unur_distr_cvec_get_pdfparams(UNUR_DISTR* distribution, double** params)

    int unur_distr_cvec_set_pdfparams_vec(UNUR_DISTR* distribution, int par, double* param_vec, int n_params)

    int unur_distr_cvec_get_pdfparams_vec(UNUR_DISTR* distribution, int par, double** param_vecs)

    int unur_distr_cvec_set_domain_rect(UNUR_DISTR* distribution, double* lowerleft, double* upperright)

    int unur_distr_cvec_is_indomain(double* x, UNUR_DISTR* distribution)

    int unur_distr_cvec_set_mode(UNUR_DISTR* distribution, double* mode)

    int unur_distr_cvec_upd_mode(UNUR_DISTR* distribution)

    double* unur_distr_cvec_get_mode(UNUR_DISTR* distribution)

    int unur_distr_cvec_set_center(UNUR_DISTR* distribution, double* center)

    double* unur_distr_cvec_get_center(UNUR_DISTR* distribution)

    int unur_distr_cvec_set_pdfvol(UNUR_DISTR* distribution, double volume)

    int unur_distr_cvec_upd_pdfvol(UNUR_DISTR* distribution)

    double unur_distr_cvec_get_pdfvol(UNUR_DISTR* distribution)

    UNUR_DISTR* unur_distr_condi_new(UNUR_DISTR* distribution, double* pos, double* dir, int k)

    int unur_distr_condi_set_condition(unur_distr* distribution, double* pos, double* dir, int k)

    int unur_distr_condi_get_condition(unur_distr* distribution, double** pos, double** dir, int* k)

    UNUR_DISTR* unur_distr_condi_get_distribution(UNUR_DISTR* distribution)

    UNUR_DISTR* unur_distr_cvemp_new(int dim)

    int unur_distr_cvemp_set_data(UNUR_DISTR* distribution, double* sample, int n_sample)

    int unur_distr_cvemp_read_data(UNUR_DISTR* distribution, char* filename)

    int unur_distr_cvemp_get_data(UNUR_DISTR* distribution, double** sample)

    UNUR_DISTR* unur_distr_discr_new()

    int unur_distr_discr_set_pv(UNUR_DISTR* distribution, double* pv, int n_pv)

    int unur_distr_discr_make_pv(UNUR_DISTR* distribution)

    int unur_distr_discr_get_pv(UNUR_DISTR* distribution, double** pv)

    int unur_distr_discr_set_pmf(UNUR_DISTR* distribution, UNUR_FUNCT_DISCR* pmf)

    int unur_distr_discr_set_cdf(UNUR_DISTR* distribution, UNUR_FUNCT_DISCR* cdf)

    int unur_distr_discr_set_invcdf(UNUR_DISTR* distribution, UNUR_IFUNCT_DISCR* invcdf)

    UNUR_FUNCT_DISCR* unur_distr_discr_get_pmf(UNUR_DISTR* distribution)

    UNUR_FUNCT_DISCR* unur_distr_discr_get_cdf(UNUR_DISTR* distribution)

    UNUR_IFUNCT_DISCR* unur_distr_discr_get_invcdf(UNUR_DISTR* distribution)

    double unur_distr_discr_eval_pv(int k, UNUR_DISTR* distribution)

    double unur_distr_discr_eval_pmf(int k, UNUR_DISTR* distribution)

    double unur_distr_discr_eval_cdf(int k, UNUR_DISTR* distribution)

    int unur_distr_discr_eval_invcdf(double u, UNUR_DISTR* distribution)

    int unur_distr_discr_set_pmfstr(UNUR_DISTR* distribution, char* pmfstr)

    int unur_distr_discr_set_cdfstr(UNUR_DISTR* distribution, char* cdfstr)

    char* unur_distr_discr_get_pmfstr(UNUR_DISTR* distribution)

    char* unur_distr_discr_get_cdfstr(UNUR_DISTR* distribution)

    int unur_distr_discr_set_pmfparams(UNUR_DISTR* distribution, double* params, int n_params)

    int unur_distr_discr_get_pmfparams(UNUR_DISTR* distribution, double** params)

    int unur_distr_discr_set_domain(UNUR_DISTR* distribution, int left, int right)

    int unur_distr_discr_get_domain(UNUR_DISTR* distribution, int* left, int* right)

    int unur_distr_discr_set_mode(UNUR_DISTR* distribution, int mode)

    int unur_distr_discr_upd_mode(UNUR_DISTR* distribution)

    int unur_distr_discr_get_mode(UNUR_DISTR* distribution)

    int unur_distr_discr_set_pmfsum(UNUR_DISTR* distribution, double sum)

    int unur_distr_discr_upd_pmfsum(UNUR_DISTR* distribution)

    double unur_distr_discr_get_pmfsum(UNUR_DISTR* distribution)

    UNUR_DISTR* unur_distr_matr_new(int n_rows, int n_cols)

    int unur_distr_matr_get_dim(UNUR_DISTR* distribution, int* n_rows, int* n_cols)

    UNUR_PAR* unur_auto_new(UNUR_DISTR* distribution)

    int unur_auto_set_logss(UNUR_PAR* parameters, int logss)

    UNUR_PAR* unur_dari_new(UNUR_DISTR* distribution)

    int unur_dari_set_squeeze(UNUR_PAR* parameters, int squeeze)

    int unur_dari_set_tablesize(UNUR_PAR* parameters, int size)

    int unur_dari_set_cpfactor(UNUR_PAR* parameters, double cp_factor)

    int unur_dari_set_verify(UNUR_PAR* parameters, int verify)

    int unur_dari_chg_verify(UNUR_GEN* generator, int verify)

    UNUR_PAR* unur_dau_new(UNUR_DISTR* distribution)

    int unur_dau_set_urnfactor(UNUR_PAR* parameters, double factor)

    UNUR_PAR* unur_dgt_new(UNUR_DISTR* distribution)

    int unur_dgt_set_guidefactor(UNUR_PAR* parameters, double factor)

    int unur_dgt_set_variant(UNUR_PAR* parameters, unsigned variant)

    int unur_dgt_eval_invcdf_recycle(UNUR_GEN* generator, double u, double* recycle)

    int unur_dgt_eval_invcdf(UNUR_GEN* generator, double u)

    UNUR_PAR* unur_dsrou_new(UNUR_DISTR* distribution)

    int unur_dsrou_set_cdfatmode(UNUR_PAR* parameters, double Fmode)

    int unur_dsrou_set_verify(UNUR_PAR* parameters, int verify)

    int unur_dsrou_chg_verify(UNUR_GEN* generator, int verify)

    int unur_dsrou_chg_cdfatmode(UNUR_GEN* generator, double Fmode)

    UNUR_PAR* unur_dss_new(UNUR_DISTR* distribution)

    UNUR_PAR* unur_arou_new(UNUR_DISTR* distribution)

    int unur_arou_set_usedars(UNUR_PAR* parameters, int usedars)

    int unur_arou_set_darsfactor(UNUR_PAR* parameters, double factor)

    int unur_arou_set_max_sqhratio(UNUR_PAR* parameters, double max_ratio)

    double unur_arou_get_sqhratio(UNUR_GEN* generator)

    double unur_arou_get_hatarea(UNUR_GEN* generator)

    double unur_arou_get_squeezearea(UNUR_GEN* generator)

    int unur_arou_set_max_segments(UNUR_PAR* parameters, int max_segs)

    int unur_arou_set_cpoints(UNUR_PAR* parameters, int n_stp, double* stp)

    int unur_arou_set_usecenter(UNUR_PAR* parameters, int usecenter)

    int unur_arou_set_guidefactor(UNUR_PAR* parameters, double factor)

    int unur_arou_set_verify(UNUR_PAR* parameters, int verify)

    int unur_arou_chg_verify(UNUR_GEN* generator, int verify)

    int unur_arou_set_pedantic(UNUR_PAR* parameters, int pedantic)

    UNUR_PAR* unur_ars_new(UNUR_DISTR* distribution)

    int unur_ars_set_max_intervals(UNUR_PAR* parameters, int max_ivs)

    int unur_ars_set_cpoints(UNUR_PAR* parameters, int n_cpoints, double* cpoints)

    int unur_ars_set_reinit_percentiles(UNUR_PAR* parameters, int n_percentiles, double* percentiles)

    int unur_ars_chg_reinit_percentiles(UNUR_GEN* generator, int n_percentiles, double* percentiles)

    int unur_ars_set_reinit_ncpoints(UNUR_PAR* parameters, int ncpoints)

    int unur_ars_chg_reinit_ncpoints(UNUR_GEN* generator, int ncpoints)

    int unur_ars_set_max_iter(UNUR_PAR* parameters, int max_iter)

    int unur_ars_set_verify(UNUR_PAR* parameters, int verify)

    int unur_ars_chg_verify(UNUR_GEN* generator, int verify)

    int unur_ars_set_pedantic(UNUR_PAR* parameters, int pedantic)

    double unur_ars_get_loghatarea(UNUR_GEN* generator)

    double unur_ars_eval_invcdfhat(UNUR_GEN* generator, double u)

    UNUR_PAR* unur_hinv_new(UNUR_DISTR* distribution)

    int unur_hinv_set_order(UNUR_PAR* parameters, int order)

    int unur_hinv_set_u_resolution(UNUR_PAR* parameters, double u_resolution)

    int unur_hinv_set_cpoints(UNUR_PAR* parameters, double* stp, int n_stp)

    int unur_hinv_set_boundary(UNUR_PAR* parameters, double left, double right)

    int unur_hinv_set_guidefactor(UNUR_PAR* parameters, double factor)

    int unur_hinv_set_max_intervals(UNUR_PAR* parameters, int max_ivs)

    int unur_hinv_get_n_intervals(UNUR_GEN* generator)

    double unur_hinv_eval_approxinvcdf(UNUR_GEN* generator, double u)

    int unur_hinv_chg_truncated(UNUR_GEN* generator, double left, double right)

    int unur_hinv_estimate_error(UNUR_GEN* generator, int samplesize, double* max_error, double* MAE)

    UNUR_PAR* unur_hrb_new(UNUR_DISTR* distribution)

    int unur_hrb_set_upperbound(UNUR_PAR* parameters, double upperbound)

    int unur_hrb_set_verify(UNUR_PAR* parameters, int verify)

    int unur_hrb_chg_verify(UNUR_GEN* generator, int verify)

    UNUR_PAR* unur_hrd_new(UNUR_DISTR* distribution)

    int unur_hrd_set_verify(UNUR_PAR* parameters, int verify)

    int unur_hrd_chg_verify(UNUR_GEN* generator, int verify)

    UNUR_PAR* unur_hri_new(UNUR_DISTR* distribution)

    int unur_hri_set_p0(UNUR_PAR* parameters, double p0)

    int unur_hri_set_verify(UNUR_PAR* parameters, int verify)

    int unur_hri_chg_verify(UNUR_GEN* generator, int verify)

    UNUR_PAR* unur_itdr_new(UNUR_DISTR* distribution)

    int unur_itdr_set_xi(UNUR_PAR* parameters, double xi)

    int unur_itdr_set_cp(UNUR_PAR* parameters, double cp)

    int unur_itdr_set_ct(UNUR_PAR* parameters, double ct)

    double unur_itdr_get_xi(UNUR_GEN* generator)

    double unur_itdr_get_cp(UNUR_GEN* generator)

    double unur_itdr_get_ct(UNUR_GEN* generator)

    double unur_itdr_get_area(UNUR_GEN* generator)

    int unur_itdr_set_verify(UNUR_PAR* parameters, int verify)

    int unur_itdr_chg_verify(UNUR_GEN* generator, int verify)

    UNUR_PAR* unur_mcorr_new(UNUR_DISTR* distribution)

    int unur_mcorr_set_eigenvalues(UNUR_PAR* par, double* eigenvalues)

    int unur_mcorr_chg_eigenvalues(UNUR_GEN* gen, double* eigenvalues)

    UNUR_PAR* unur_ninv_new(UNUR_DISTR* distribution)

    int unur_ninv_set_useregula(UNUR_PAR* parameters)

    int unur_ninv_set_usenewton(UNUR_PAR* parameters)

    int unur_ninv_set_usebisect(UNUR_PAR* parameters)

    int unur_ninv_set_max_iter(UNUR_PAR* parameters, int max_iter)

    int unur_ninv_chg_max_iter(UNUR_GEN* generator, int max_iter)

    int unur_ninv_set_x_resolution(UNUR_PAR* parameters, double x_resolution)

    int unur_ninv_chg_x_resolution(UNUR_GEN* generator, double x_resolution)

    int unur_ninv_set_u_resolution(UNUR_PAR* parameters, double u_resolution)

    int unur_ninv_chg_u_resolution(UNUR_GEN* generator, double u_resolution)

    int unur_ninv_set_start(UNUR_PAR* parameters, double left, double right)

    int unur_ninv_chg_start(UNUR_GEN* gen, double left, double right)

    int unur_ninv_set_table(UNUR_PAR* parameters, int no_of_points)

    int unur_ninv_chg_table(UNUR_GEN* gen, int no_of_points)

    int unur_ninv_chg_truncated(UNUR_GEN* gen, double left, double right)

    double unur_ninv_eval_approxinvcdf(UNUR_GEN* generator, double u)

    UNUR_PAR* unur_nrou_new(UNUR_DISTR* distribution)

    int unur_nrou_set_u(UNUR_PAR* parameters, double umin, double umax)

    int unur_nrou_set_v(UNUR_PAR* parameters, double vmax)

    int unur_nrou_set_r(UNUR_PAR* parameters, double r)

    int unur_nrou_set_center(UNUR_PAR* parameters, double center)

    int unur_nrou_set_verify(UNUR_PAR* parameters, int verify)

    int unur_nrou_chg_verify(UNUR_GEN* generator, int verify)

    UNUR_PAR* unur_pinv_new(UNUR_DISTR* distribution)

    int unur_pinv_set_order(UNUR_PAR* parameters, int order)

    int unur_pinv_set_smoothness(UNUR_PAR* parameters, int smoothness)

    int unur_pinv_set_u_resolution(UNUR_PAR* parameters, double u_resolution)

    int unur_pinv_set_use_upoints(UNUR_PAR* parameters, int use_upoints)

    int unur_pinv_set_usepdf(UNUR_PAR* parameters)

    int unur_pinv_set_usecdf(UNUR_PAR* parameters)

    int unur_pinv_set_boundary(UNUR_PAR* parameters, double left, double right)

    int unur_pinv_set_searchboundary(UNUR_PAR* parameters, int left, int right)

    int unur_pinv_set_max_intervals(UNUR_PAR* parameters, int max_ivs)

    int unur_pinv_get_n_intervals(UNUR_GEN* generator)

    int unur_pinv_set_keepcdf(UNUR_PAR* parameters, int keepcdf)

    double unur_pinv_eval_approxinvcdf(UNUR_GEN* generator, double u)

    double unur_pinv_eval_approxcdf(UNUR_GEN* generator, double x)

    int unur_pinv_estimate_error(UNUR_GEN* generator, int samplesize, double* max_error, double* MAE)

    UNUR_PAR* unur_srou_new(UNUR_DISTR* distribution)

    int unur_srou_set_r(UNUR_PAR* parameters, double r)

    int unur_srou_set_cdfatmode(UNUR_PAR* parameters, double Fmode)

    int unur_srou_set_pdfatmode(UNUR_PAR* parameters, double fmode)

    int unur_srou_set_usesqueeze(UNUR_PAR* parameters, int usesqueeze)

    int unur_srou_set_usemirror(UNUR_PAR* parameters, int usemirror)

    int unur_srou_set_verify(UNUR_PAR* parameters, int verify)

    int unur_srou_chg_verify(UNUR_GEN* generator, int verify)

    int unur_srou_chg_cdfatmode(UNUR_GEN* generator, double Fmode)

    int unur_srou_chg_pdfatmode(UNUR_GEN* generator, double fmode)

    UNUR_PAR* unur_ssr_new(UNUR_DISTR* distribution)

    int unur_ssr_set_cdfatmode(UNUR_PAR* parameters, double Fmode)

    int unur_ssr_set_pdfatmode(UNUR_PAR* parameters, double fmode)

    int unur_ssr_set_usesqueeze(UNUR_PAR* parameters, int usesqueeze)

    int unur_ssr_set_verify(UNUR_PAR* parameters, int verify)

    int unur_ssr_chg_verify(UNUR_GEN* generator, int verify)

    int unur_ssr_chg_cdfatmode(UNUR_GEN* generator, double Fmode)

    int unur_ssr_chg_pdfatmode(UNUR_GEN* generator, double fmode)

    UNUR_PAR* unur_tabl_new(UNUR_DISTR* distribution)

    int unur_tabl_set_variant_ia(UNUR_PAR* parameters, int use_ia)

    int unur_tabl_set_cpoints(UNUR_PAR* parameters, int n_cpoints, double* cpoints)

    int unur_tabl_set_nstp(UNUR_PAR* parameters, int n_stp)

    int unur_tabl_set_useear(UNUR_PAR* parameters, int useear)

    int unur_tabl_set_areafraction(UNUR_PAR* parameters, double fraction)

    int unur_tabl_set_usedars(UNUR_PAR* parameters, int usedars)

    int unur_tabl_set_darsfactor(UNUR_PAR* parameters, double factor)

    int unur_tabl_set_variant_splitmode(UNUR_PAR* parameters, unsigned splitmode)

    int unur_tabl_set_max_sqhratio(UNUR_PAR* parameters, double max_ratio)

    double unur_tabl_get_sqhratio(UNUR_GEN* generator)

    double unur_tabl_get_hatarea(UNUR_GEN* generator)

    double unur_tabl_get_squeezearea(UNUR_GEN* generator)

    int unur_tabl_set_max_intervals(UNUR_PAR* parameters, int max_ivs)

    int unur_tabl_get_n_intervals(UNUR_GEN* generator)

    int unur_tabl_set_slopes(UNUR_PAR* parameters, double* slopes, int n_slopes)

    int unur_tabl_set_guidefactor(UNUR_PAR* parameters, double factor)

    int unur_tabl_set_boundary(UNUR_PAR* parameters, double left, double right)

    int unur_tabl_chg_truncated(UNUR_GEN* gen, double left, double right)

    int unur_tabl_set_verify(UNUR_PAR* parameters, int verify)

    int unur_tabl_chg_verify(UNUR_GEN* generator, int verify)

    int unur_tabl_set_pedantic(UNUR_PAR* parameters, int pedantic)

    UNUR_PAR* unur_tdr_new(UNUR_DISTR* distribution)

    int unur_tdr_set_c(UNUR_PAR* parameters, double c)

    int unur_tdr_set_variant_gw(UNUR_PAR* parameters)

    int unur_tdr_set_variant_ps(UNUR_PAR* parameters)

    int unur_tdr_set_variant_ia(UNUR_PAR* parameters)

    int unur_tdr_set_usedars(UNUR_PAR* parameters, int usedars)

    int unur_tdr_set_darsfactor(UNUR_PAR* parameters, double factor)

    int unur_tdr_set_cpoints(UNUR_PAR* parameters, int n_stp, double* stp)

    int unur_tdr_set_reinit_percentiles(UNUR_PAR* parameters, int n_percentiles, double* percentiles)

    int unur_tdr_chg_reinit_percentiles(UNUR_GEN* generator, int n_percentiles, double* percentiles)

    int unur_tdr_set_reinit_ncpoints(UNUR_PAR* parameters, int ncpoints)

    int unur_tdr_chg_reinit_ncpoints(UNUR_GEN* generator, int ncpoints)

    int unur_tdr_chg_truncated(UNUR_GEN* gen, double left, double right)

    int unur_tdr_set_max_sqhratio(UNUR_PAR* parameters, double max_ratio)

    double unur_tdr_get_sqhratio(UNUR_GEN* generator)

    double unur_tdr_get_hatarea(UNUR_GEN* generator)

    double unur_tdr_get_squeezearea(UNUR_GEN* generator)

    int unur_tdr_set_max_intervals(UNUR_PAR* parameters, int max_ivs)

    int unur_tdr_set_usecenter(UNUR_PAR* parameters, int usecenter)

    int unur_tdr_set_usemode(UNUR_PAR* parameters, int usemode)

    int unur_tdr_set_guidefactor(UNUR_PAR* parameters, double factor)

    int unur_tdr_set_verify(UNUR_PAR* parameters, int verify)

    int unur_tdr_chg_verify(UNUR_GEN* generator, int verify)

    int unur_tdr_set_pedantic(UNUR_PAR* parameters, int pedantic)

    double unur_tdr_eval_invcdfhat(UNUR_GEN* generator, double u, double* hx, double* fx, double* sqx)

    int _unur_tdr_is_ARS_running(UNUR_GEN* generator)

    UNUR_PAR* unur_utdr_new(UNUR_DISTR* distribution)

    int unur_utdr_set_pdfatmode(UNUR_PAR* parameters, double fmode)

    int unur_utdr_set_cpfactor(UNUR_PAR* parameters, double cp_factor)

    int unur_utdr_set_deltafactor(UNUR_PAR* parameters, double delta)

    int unur_utdr_set_verify(UNUR_PAR* parameters, int verify)

    int unur_utdr_chg_verify(UNUR_GEN* generator, int verify)

    int unur_utdr_chg_pdfatmode(UNUR_GEN* generator, double fmode)

    UNUR_PAR* unur_empk_new(UNUR_DISTR* distribution)

    int unur_empk_set_kernel(UNUR_PAR* parameters, unsigned kernel)

    int unur_empk_set_kernelgen(UNUR_PAR* parameters, UNUR_GEN* kernelgen, double alpha, double kernelvar)

    int unur_empk_set_beta(UNUR_PAR* parameters, double beta)

    int unur_empk_set_smoothing(UNUR_PAR* parameters, double smoothing)

    int unur_empk_chg_smoothing(UNUR_GEN* generator, double smoothing)

    int unur_empk_set_varcor(UNUR_PAR* parameters, int varcor)

    int unur_empk_chg_varcor(UNUR_GEN* generator, int varcor)

    int unur_empk_set_positive(UNUR_PAR* parameters, int positive)

    UNUR_PAR* unur_empl_new(UNUR_DISTR* distribution)

    UNUR_PAR* unur_hist_new(UNUR_DISTR* distribution)

    UNUR_PAR* unur_mvtdr_new(UNUR_DISTR* distribution)

    int unur_mvtdr_set_stepsmin(UNUR_PAR* parameters, int stepsmin)

    int unur_mvtdr_set_boundsplitting(UNUR_PAR* parameters, double boundsplitting)

    int unur_mvtdr_set_maxcones(UNUR_PAR* parameters, int maxcones)

    int unur_mvtdr_get_ncones(UNUR_GEN* generator)

    double unur_mvtdr_get_hatvol(UNUR_GEN* generator)

    int unur_mvtdr_set_verify(UNUR_PAR* parameters, int verify)

    int unur_mvtdr_chg_verify(UNUR_GEN* generator, int verify)

    UNUR_PAR* unur_norta_new(UNUR_DISTR* distribution)

    UNUR_PAR* unur_vempk_new(UNUR_DISTR* distribution)

    int unur_vempk_set_smoothing(UNUR_PAR* parameters, double smoothing)

    int unur_vempk_chg_smoothing(UNUR_GEN* generator, double smoothing)

    int unur_vempk_set_varcor(UNUR_PAR* parameters, int varcor)

    int unur_vempk_chg_varcor(UNUR_GEN* generator, int varcor)

    UNUR_PAR* unur_vnrou_new(UNUR_DISTR* distribution)

    int unur_vnrou_set_u(UNUR_PAR* parameters, double* umin, double* umax)

    int unur_vnrou_chg_u(UNUR_GEN* generator, double* umin, double* umax)

    int unur_vnrou_set_v(UNUR_PAR* parameters, double vmax)

    int unur_vnrou_chg_v(UNUR_GEN* generator, double vmax)

    int unur_vnrou_set_r(UNUR_PAR* parameters, double r)

    int unur_vnrou_set_verify(UNUR_PAR* parameters, int verify)

    int unur_vnrou_chg_verify(UNUR_GEN* generator, int verify)

    double unur_vnrou_get_volumehat(UNUR_GEN* generator)

    UNUR_PAR* unur_gibbs_new(UNUR_DISTR* distribution)

    int unur_gibbs_set_variant_coordinate(UNUR_PAR* parameters)

    int unur_gibbs_set_variant_random_direction(UNUR_PAR* parameters)

    int unur_gibbs_set_c(UNUR_PAR* parameters, double c)

    int unur_gibbs_set_startingpoint(UNUR_PAR* parameters, double* x0)

    int unur_gibbs_set_thinning(UNUR_PAR* parameters, int thinning)

    int unur_gibbs_set_burnin(UNUR_PAR* parameters, int burnin)

    double* unur_gibbs_get_state(UNUR_GEN* generator)

    int unur_gibbs_chg_state(UNUR_GEN* generator, double* state)

    int unur_gibbs_reset_state(UNUR_GEN* generator)

    UNUR_PAR* unur_hitro_new(UNUR_DISTR* distribution)

    int unur_hitro_set_variant_coordinate(UNUR_PAR* parameters)

    int unur_hitro_set_variant_random_direction(UNUR_PAR* parameters)

    int unur_hitro_set_use_adaptiveline(UNUR_PAR* parameters, int adaptive)

    int unur_hitro_set_use_boundingrectangle(UNUR_PAR* parameters, int rectangle)

    int unur_hitro_set_use_adaptiverectangle(UNUR_PAR* parameters, int adaptive)

    int unur_hitro_set_r(UNUR_PAR* parameters, double r)

    int unur_hitro_set_v(UNUR_PAR* parameters, double vmax)

    int unur_hitro_set_u(UNUR_PAR* parameters, double* umin, double* umax)

    int unur_hitro_set_adaptive_multiplier(UNUR_PAR* parameters, double factor)

    int unur_hitro_set_startingpoint(UNUR_PAR* parameters, double* x0)

    int unur_hitro_set_thinning(UNUR_PAR* parameters, int thinning)

    int unur_hitro_set_burnin(UNUR_PAR* parameters, int burnin)

    double* unur_hitro_get_state(UNUR_GEN* generator)

    int unur_hitro_chg_state(UNUR_GEN* generator, double* state)

    int unur_hitro_reset_state(UNUR_GEN* generator)

    UNUR_PAR* unur_cstd_new(UNUR_DISTR* distribution)

    int unur_cstd_set_variant(UNUR_PAR* parameters, unsigned variant)

    int unur_cstd_chg_truncated(UNUR_GEN* generator, double left, double right)

    double unur_cstd_eval_invcdf(UNUR_GEN* generator, double u)

    UNUR_PAR* unur_dstd_new(UNUR_DISTR* distribution)

    int unur_dstd_set_variant(UNUR_PAR* parameters, unsigned variant)

    int unur_dstd_chg_truncated(UNUR_GEN* generator, int left, int right)

    int unur_dstd_eval_invcdf(UNUR_GEN* generator, double u)

    UNUR_PAR* unur_mvstd_new(UNUR_DISTR* distribution)

    UNUR_PAR* unur_mixt_new(int n, double* prob, UNUR_GEN** comp)

    int unur_mixt_set_useinversion(UNUR_PAR* parameters, int useinv)

    double unur_mixt_eval_invcdf(UNUR_GEN* generator, double u)

    UNUR_PAR* unur_cext_new(UNUR_DISTR* distribution)

    ctypedef int (*_unur_cext_set_init_init_ft)(UNUR_GEN* gen)

    int unur_cext_set_init(UNUR_PAR* parameters, _unur_cext_set_init_init_ft init)

    ctypedef double (*_unur_cext_set_sample_sample_ft)(UNUR_GEN* gen)

    int unur_cext_set_sample(UNUR_PAR* parameters, _unur_cext_set_sample_sample_ft sample)

    void* unur_cext_get_params(UNUR_GEN* generator, size_t size)

    double* unur_cext_get_distrparams(UNUR_GEN* generator)

    int unur_cext_get_ndistrparams(UNUR_GEN* generator)

    UNUR_PAR* unur_dext_new(UNUR_DISTR* distribution)

    ctypedef int (*_unur_dext_set_init_init_ft)(UNUR_GEN* gen)

    int unur_dext_set_init(UNUR_PAR* parameters, _unur_dext_set_init_init_ft init)

    ctypedef int (*_unur_dext_set_sample_sample_ft)(UNUR_GEN* gen)

    int unur_dext_set_sample(UNUR_PAR* parameters, _unur_dext_set_sample_sample_ft sample)

    void* unur_dext_get_params(UNUR_GEN* generator, size_t size)

    double* unur_dext_get_distrparams(UNUR_GEN* generator)

    int unur_dext_get_ndistrparams(UNUR_GEN* generator)

    UNUR_PAR* unur_unif_new(UNUR_DISTR* dummy)

    UNUR_GEN* unur_str2gen(char* string)

    UNUR_DISTR* unur_str2distr(char* string)

    UNUR_GEN* unur_makegen_ssu(char* distrstr, char* methodstr, UNUR_URNG* urng)

    UNUR_GEN* unur_makegen_dsu(UNUR_DISTR* distribution, char* methodstr, UNUR_URNG* urng)

    UNUR_PAR* _unur_str2par(UNUR_DISTR* distribution, char* method, unur_slist** mlist)

    UNUR_GEN* unur_init(UNUR_PAR* parameters)

    int unur_reinit(UNUR_GEN* generator)

    int unur_sample_discr(UNUR_GEN* generator)

    double unur_sample_cont(UNUR_GEN* generator)

    int unur_sample_vec(UNUR_GEN* generator, double* vector)

    int unur_sample_matr(UNUR_GEN* generator, double* matrix)

    double unur_quantile(UNUR_GEN* generator, double U)

    void unur_free(UNUR_GEN* generator)

    char* unur_gen_info(UNUR_GEN* generator, int help)

    int unur_get_dimension(UNUR_GEN* generator)

    char* unur_get_genid(UNUR_GEN* generator)

    UNUR_DISTR* unur_get_distr(UNUR_GEN* generator)

    int unur_set_use_distr_privatecopy(UNUR_PAR* parameters, int use_privatecopy)

    UNUR_GEN* unur_gen_clone(UNUR_GEN* gen)

    void unur_par_free(UNUR_PAR* par)

    cdef enum:
        UNUR_DISTR_GENERIC
        UNUR_DISTR_CORDER
        UNUR_DISTR_CXTRANS
        UNUR_DISTR_CONDI
        UNUR_DISTR_BETA
        UNUR_DISTR_CAUCHY
        UNUR_DISTR_CHI
        UNUR_DISTR_CHISQUARE
        UNUR_DISTR_EPANECHNIKOV
        UNUR_DISTR_EXPONENTIAL
        UNUR_DISTR_EXTREME_I
        UNUR_DISTR_EXTREME_II
        UNUR_DISTR_F
        UNUR_DISTR_GAMMA
        UNUR_DISTR_GHYP
        UNUR_DISTR_GIG
        UNUR_DISTR_GIG2
        UNUR_DISTR_HYPERBOLIC
        UNUR_DISTR_IG
        UNUR_DISTR_LAPLACE
        UNUR_DISTR_LOGISTIC
        UNUR_DISTR_LOGNORMAL
        UNUR_DISTR_LOMAX
        UNUR_DISTR_NORMAL
        UNUR_DISTR_GAUSSIAN
        UNUR_DISTR_PARETO
        UNUR_DISTR_POWEREXPONENTIAL
        UNUR_DISTR_RAYLEIGH
        UNUR_DISTR_SLASH
        UNUR_DISTR_STUDENT
        UNUR_DISTR_TRIANGULAR
        UNUR_DISTR_UNIFORM
        UNUR_DISTR_BOXCAR
        UNUR_DISTR_WEIBULL
        UNUR_DISTR_BURR_I
        UNUR_DISTR_BURR_II
        UNUR_DISTR_BURR_III
        UNUR_DISTR_BURR_IV
        UNUR_DISTR_BURR_V
        UNUR_DISTR_BURR_VI
        UNUR_DISTR_BURR_VII
        UNUR_DISTR_BURR_VIII
        UNUR_DISTR_BURR_IX
        UNUR_DISTR_BURR_X
        UNUR_DISTR_BURR_XI
        UNUR_DISTR_BURR_XII
        UNUR_DISTR_BINOMIAL
        UNUR_DISTR_GEOMETRIC
        UNUR_DISTR_HYPERGEOMETRIC
        UNUR_DISTR_LOGARITHMIC
        UNUR_DISTR_NEGATIVEBINOMIAL
        UNUR_DISTR_POISSON
        UNUR_DISTR_ZIPF
        UNUR_DISTR_MCAUCHY
        UNUR_DISTR_MNORMAL
        UNUR_DISTR_MSTUDENT
        UNUR_DISTR_MEXPONENTIAL
        UNUR_DISTR_COPULA
        UNUR_DISTR_MCORRELATION

    UNUR_DISTR* unur_distr_beta(double* params, int n_params)

    UNUR_DISTR* unur_distr_burr(double* params, int n_params)

    UNUR_DISTR* unur_distr_cauchy(double* params, int n_params)

    UNUR_DISTR* unur_distr_chi(double* params, int n_params)

    UNUR_DISTR* unur_distr_chisquare(double* params, int n_params)

    UNUR_DISTR* unur_distr_exponential(double* params, int n_params)

    UNUR_DISTR* unur_distr_extremeI(double* params, int n_params)

    UNUR_DISTR* unur_distr_extremeII(double* params, int n_params)

    UNUR_DISTR* unur_distr_F(double* params, int n_params)

    UNUR_DISTR* unur_distr_gamma(double* params, int n_params)

    UNUR_DISTR* unur_distr_ghyp(double* params, int n_params)

    UNUR_DISTR* unur_distr_gig(double* params, int n_params)

    UNUR_DISTR* unur_distr_gig2(double* params, int n_params)

    UNUR_DISTR* unur_distr_hyperbolic(double* params, int n_params)

    UNUR_DISTR* unur_distr_ig(double* params, int n_params)

    UNUR_DISTR* unur_distr_laplace(double* params, int n_params)

    UNUR_DISTR* unur_distr_logistic(double* params, int n_params)

    UNUR_DISTR* unur_distr_lognormal(double* params, int n_params)

    UNUR_DISTR* unur_distr_lomax(double* params, int n_params)

    UNUR_DISTR* unur_distr_normal(double* params, int n_params)

    UNUR_DISTR* unur_distr_pareto(double* params, int n_params)

    UNUR_DISTR* unur_distr_powerexponential(double* params, int n_params)

    UNUR_DISTR* unur_distr_rayleigh(double* params, int n_params)

    UNUR_DISTR* unur_distr_slash(double* params, int n_params)

    UNUR_DISTR* unur_distr_student(double* params, int n_params)

    UNUR_DISTR* unur_distr_triangular(double* params, int n_params)

    UNUR_DISTR* unur_distr_uniform(double* params, int n_params)

    UNUR_DISTR* unur_distr_weibull(double* params, int n_params)

    UNUR_DISTR* unur_distr_multinormal(int dim, double* mean, double* covar)

    UNUR_DISTR* unur_distr_multicauchy(int dim, double* mean, double* covar)

    UNUR_DISTR* unur_distr_multistudent(int dim, double nu, double* mean, double* covar)

    UNUR_DISTR* unur_distr_multiexponential(int dim, double* sigma, double* theta)

    UNUR_DISTR* unur_distr_copula(int dim, double* rankcorr)

    UNUR_DISTR* unur_distr_correlation(int n)

    UNUR_DISTR* unur_distr_binomial(double* params, int n_params)

    UNUR_DISTR* unur_distr_geometric(double* params, int n_params)

    UNUR_DISTR* unur_distr_hypergeometric(double* params, int n_params)

    UNUR_DISTR* unur_distr_logarithmic(double* params, int n_params)

    UNUR_DISTR* unur_distr_negativebinomial(double* params, int n_params)

    UNUR_DISTR* unur_distr_poisson(double* params, int n_params)

    UNUR_DISTR* unur_distr_zipf(double* params, int n_params)

    FILE* unur_set_stream(FILE* new_stream)

    FILE* unur_get_stream()

    int unur_set_debug(UNUR_PAR* parameters, unsigned debug)

    int unur_chg_debug(UNUR_GEN* generator, unsigned debug)

    int unur_set_default_debug(unsigned debug)

    int unur_errno

    int unur_get_errno()

    void unur_reset_errno()

    char* unur_get_strerror(int errnocode)

    UNUR_ERROR_HANDLER* unur_set_error_handler(UNUR_ERROR_HANDLER* new_handler)

    UNUR_ERROR_HANDLER* unur_set_error_handler_off()

    cdef enum:
        UNUR_SUCCESS
        UNUR_FAILURE
        UNUR_ERR_DISTR_SET
        UNUR_ERR_DISTR_GET
        UNUR_ERR_DISTR_NPARAMS
        UNUR_ERR_DISTR_DOMAIN
        UNUR_ERR_DISTR_GEN
        UNUR_ERR_DISTR_REQUIRED
        UNUR_ERR_DISTR_UNKNOWN
        UNUR_ERR_DISTR_INVALID
        UNUR_ERR_DISTR_DATA
        UNUR_ERR_DISTR_PROP
        UNUR_ERR_PAR_SET
        UNUR_ERR_PAR_VARIANT
        UNUR_ERR_PAR_INVALID
        UNUR_ERR_GEN
        UNUR_ERR_GEN_DATA
        UNUR_ERR_GEN_CONDITION
        UNUR_ERR_GEN_INVALID
        UNUR_ERR_GEN_SAMPLING
        UNUR_ERR_NO_REINIT
        UNUR_ERR_NO_QUANTILE
        UNUR_ERR_URNG
        UNUR_ERR_URNG_MISS
        UNUR_ERR_STR
        UNUR_ERR_STR_UNKNOWN
        UNUR_ERR_STR_SYNTAX
        UNUR_ERR_STR_INVALID
        UNUR_ERR_FSTR_SYNTAX
        UNUR_ERR_FSTR_DERIV
        UNUR_ERR_DOMAIN
        UNUR_ERR_ROUNDOFF
        UNUR_ERR_MALLOC
        UNUR_ERR_NULL
        UNUR_ERR_COOKIE
        UNUR_ERR_GENERIC
        UNUR_ERR_SILENT
        UNUR_ERR_INF
        UNUR_ERR_NAN
        UNUR_ERR_COMPILE
        UNUR_ERR_SHOULD_NOT_HAPPEN

    double INFINITY

    unur_slist* _unur_slist_new()

    int _unur_slist_append(unur_slist* slist, void* element)

    int _unur_slist_length(unur_slist* slist)

    void* _unur_slist_get(unur_slist* slist, int n)

    void* _unur_slist_replace(unur_slist* slist, int n, void* element)

    void _unur_slist_free(unur_slist* slist)
