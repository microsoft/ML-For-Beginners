from io import open
import os
import re

import numpy as np

from .parse_jmulti_vecm_output import stringify_var_names, sublists

debug_mode = False

here = os.path.dirname(os.path.realpath(__file__))


def print_debug_output(results, dt):
    print("\n\n\nDETERMINISTIC TERMS: " + dt)
    coefs = results["est"]["Lagged endogenous term"]
    print("coefs:")
    print(str(type(coefs)) + str(coefs.shape))
    print(coefs)
    print("se: ")
    print(results["se"]["Lagged endogenous term"])
    print("t: ")
    print(results["t"]["Lagged endogenous term"])
    print("p: ")
    print(results["p"]["Lagged endogenous term"])


def dt_s_tup_to_string(dt_s_tup):
    """

    Parameters
    ----------
    dt_s_tup : tuple
        A tuple of length 2.
        The first entry is a string specifying the deterministic term without
        any information about seasonal terms (for example "nc" or "c").
        The second entry is an int specifying the number of seasons.

    Returns
    -------
    dt_string : str
        Returns dt_s_tup[0], if dt_s_tup[1] is 0 (i.e. no seasons).
        If dt_s_tup[1] is > 0 (i.e. there are seasons) add an "s" to the string
        in dt_s_tup[0] like in the following examples:
        "nc" --> "ncs"
        "c" --> "cs"
        "ct" --> "cst"
    """
    dt_string = dt_s_tup[0]  # string for identifying the file to parse.
    if dt_s_tup[1] > 0:  # if there are seasons in the model
        if dt_string == "nc":
            dt_string = dt_string[:2] + "s"
        if dt_string == "c" or dt_string == "ct":
            dt_string = dt_string[:1] + "s" + dt_string[1:]
    return dt_string


def load_results_jmulti(dataset, dt_s_list):
    """

    Parameters
    ----------
    dataset : module
        A data module in the statsmodels/datasets directory that defines a
        __str__() method returning the dataset's name.
    dt_s_list : list
        A list of strings where each string represents a combination of
        deterministic terms.

    Returns
    -------
    result : dict
        A dict (keys: tuples of deterministic terms and seasonal terms)
        of dicts (keys: strings "est" (for estimators),
                              "se" (for standard errors),
                              "t" (for t-values),
                              "p" (for p-values))
        of dicts (keys: strings "alpha", "beta", "Gamma" and other results)
    """
    source = "jmulti"

    results_dict_per_det_terms = dict.fromkeys(dt_s_list)

    for dt_s in dt_s_list:
        dt_string = dt_s_tup_to_string(dt_s)
        params_file = dataset.__str__()+"_"+source+"_"+dt_string+".txt"
        params_file = os.path.join(here, params_file)
        # sections in jmulti output:
        section_headers = ["Lagged endogenous term",  # parameter matrices
                           "Deterministic term"]      # c, s, ct
        if dt_string == "nc":
            del section_headers[-1]

        results = dict()
        results["est"] = dict.fromkeys(section_headers)
        results["se"] = dict.fromkeys(section_headers)
        results["t"] = dict.fromkeys(section_headers)
        results["p"] = dict.fromkeys(section_headers)
        result = []
        result_se = []
        result_t = []
        result_p = []

        rows = 0
        started_reading_section = False
        start_end_mark = "-----"

        # ---------------------------------------------------------------------
        # parse information about \alpha, \beta, \Gamma, deterministic of VECM
        # and A_i and deterministic of corresponding VAR:
        section = -1
        params_file = open(params_file, encoding='latin_1')
        for line in params_file:
            if section == -1 and section_headers[section+1] not in line:
                continue
            if section < len(section_headers)-1 \
                    and section_headers[section+1] in line:  # new section
                section += 1
                continue
            if not started_reading_section:
                if line.startswith(start_end_mark):
                    started_reading_section = True
                continue
            if started_reading_section:
                if line.startswith(start_end_mark):
                    if result == []:  # no values collected in section "Legend"
                        started_reading_section = False
                        continue
                    results["est"][section_headers[section]] = np.column_stack(
                            result)
                    result = []
                    results["se"][section_headers[section]] = np.column_stack(
                            result_se)
                    result_se = []
                    results["t"][section_headers[section]] = np.column_stack(
                            result_t)
                    result_t = []
                    results["p"][section_headers[section]] = np.column_stack(
                            result_p)
                    result_p = []
                    started_reading_section = False
                    continue
                str_number = r"-?\d+\.\d{3}"
                regex_est = re.compile(str_number + r"[^\)\]\}]")
                est_col = re.findall(regex_est, line)
                # standard errors in parantheses in JMulTi output:
                regex_se = re.compile(r"\(" + str_number + r"\)")
                se_col = re.findall(regex_se, line)
                # t-values in brackets in JMulTi output:
                regex_t_value = re.compile(r"\[" + str_number + r"\]")
                t_col = re.findall(regex_t_value, line)
                # p-values in braces in JMulTi output:
                regex_p_value = re.compile(r"\{" + str_number + r"\}")
                p_col = re.findall(regex_p_value, line)
                if result == [] and est_col != []:
                    rows = len(est_col)
                if est_col != []:
                    est_col = [float(el) for el in est_col]
                    result.append(est_col)
                elif se_col != []:
                    for i in range(rows):
                        se_col[i] = se_col[i].replace("(", "").replace(")", "")
                    se_col = [float(el) for el in se_col]
                    result_se.append(se_col)
                elif t_col != []:
                    for i in range(rows):
                        t_col[i] = t_col[i].replace("[", "").replace("]", "")
                    t_col = [float(el) for el in t_col]
                    result_t.append(t_col)
                elif p_col != []:
                    for i in range(rows):
                        p_col[i] = p_col[i].replace("{", "").replace("}", "")
                    p_col = [float(el) for el in p_col]
                    result_p.append(p_col)
        params_file.close()

        # ---------------------------------------------------------------------
        # parse information regarding \Sigma_u
        sigmau_file = dataset.__str__() + "_" + source + "_" + dt_string \
            + "_Sigmau" + ".txt"
        sigmau_file = os.path.join(here, sigmau_file)
        rows_to_parse = 0
        # all numbers of Sigma_u in notation with e (e.g. 2.283862e-05)
        regex_est = re.compile(r"\s+\S+e\S+")
        sigmau_section_reached = False
        sigmau_file = open(sigmau_file, encoding='latin_1')
        for line in sigmau_file:
            if line.startswith("Log Likelihood:"):
                line = line[len("Log Likelihood:"):]
                results["log_like"] = float(re.findall(regex_est, line)[0])
                continue
            if not sigmau_section_reached and "Covariance:" not in line:
                continue
            if "Covariance:" in line:
                sigmau_section_reached = True
                row = re.findall(regex_est, line)
                rows_to_parse = len(row)  # Sigma_u quadratic ==> #rows==#cols
                sigma_u = np.empty((rows_to_parse, rows_to_parse))
            row = re.findall(regex_est, line)
            rows_to_parse -= 1
            sigma_u[rows_to_parse] = row  # rows are added in reverse order...
            if rows_to_parse == 0:
                break
        sigmau_file.close()
        results["est"]["Sigma_u"] = sigma_u[::-1]  # ...and reversed again here

        # ---------------------------------------------------------------------
        # parse forecast related output:
        fc_file = dataset.__str__() + "_" + source + "_" + dt_string \
            + "_fc5" + ".txt"
        fc_file = os.path.join(here, fc_file)
        fc, lower, upper, plu_min = [], [], [], []
        fc_file = open(fc_file, encoding='latin_1')
        for line in fc_file:
            str_number = r"(\s+-?\d+\.\d{3}\s*)"
            regex_number = re.compile(str_number)
            numbers = re.findall(regex_number, line)
            if numbers == []:
                continue
            fc.append(float(numbers[0]))
            lower.append(float(numbers[1]))
            upper.append(float(numbers[2]))
            plu_min.append(float(numbers[3]))
        fc_file.close()
        neqs = len(results["est"]["Sigma_u"])
        fc = np.hstack(np.vsplit(np.array(fc)[:, None], neqs))
        lower = np.hstack(np.vsplit(np.array(lower)[:, None], neqs))
        upper = np.hstack(np.vsplit(np.array(upper)[:, None], neqs))
        results["fc"] = dict.fromkeys(["fc", "lower", "upper"])
        results["fc"]["fc"] = fc
        results["fc"]["lower"] = lower
        results["fc"]["upper"] = upper

        # ---------------------------------------------------------------------
        # parse output related to Granger-caus. and instantaneous causality:
        results["granger_caus"] = dict.fromkeys(["p", "test_stat"])
        results["granger_caus"]["p"] = dict()
        results["granger_caus"]["test_stat"] = dict()
        results["inst_caus"] = dict.fromkeys(["p", "test_stat"])
        results["inst_caus"]["p"] = dict()
        results["inst_caus"]["test_stat"] = dict()
        vn = dataset.variable_names
        # all possible combinations of potentially causing variables
        # (at least 1 variable and not all variables together):
        var_combs = sublists(vn, 1, len(vn)-1)
        if debug_mode:
            print("\n\n\n" + dt_string)
        for causing in var_combs:
            caused = tuple(name for name in vn if name not in causing)
            causality_file = dataset.__str__() + "_" + source + "_" \
                + dt_string + "_granger_causality_" \
                + stringify_var_names(causing, "_") + ".txt"
            causality_file = os.path.join(here, causality_file)
            causality_file = open(causality_file, encoding="latin_1")
            causality_results = []
            for line in causality_file:
                str_number = r"\d+\.\d{4}"
                regex_number = re.compile(str_number)
                number = re.search(regex_number, line)
                if number is None:
                    continue
                number = float(number.group(0))
                causality_results.append(number)
            causality_file.close()
            results["granger_caus"]["test_stat"][(causing, caused)] = \
                causality_results[0]
            results["granger_caus"]["p"][(causing, caused)] =\
                causality_results[1]
            results["inst_caus"]["test_stat"][(causing, caused)] = \
                causality_results[2]
            results["inst_caus"]["p"][(causing, caused)] = \
                causality_results[3]

        # ---------------------------------------------------------------------
        # parse output related to impulse-response analysis:
        ir_file = dataset.__str__() + "_" + source + "_" + dt_string \
            + "_ir" + ".txt"
        ir_file = os.path.join(here, ir_file)
        ir_file = open(ir_file, encoding='latin_1')
        causing = None
        caused = None
        data = None
        regex_vars = re.compile(r"\w+")
        regex_vals = re.compile(r"-?\d+\.\d{4}")
        line_start_causing = "time"
        data_line_indicator = "point estimate"
        data_rows_read = 0
        for line in ir_file:
            if causing is None and not line.startswith(line_start_causing):
                continue  # no relevant info in the header
            if line.startswith(line_start_causing):
                line = line[4:]
                causing = re.findall(regex_vars, line)
                # 21 periods shown in JMulTi output
                data = np.empty((21, len(causing)))
                continue
            if caused is None:
                caused = re.findall(regex_vars, line)
                continue
            # now start collecting the values:
            if data_line_indicator not in line:
                continue
            start = line.find(data_line_indicator) + len(data_line_indicator)
            line = line[start:]
            data[data_rows_read] = re.findall(regex_vals, line)
            data_rows_read += 1
        ir_file.close()
        results["ir"] = data

        # ---------------------------------------------------------------------
        # parse output related to lag order selection:
        lagorder_file = dataset.__str__() + "_" + source + "_" + dt_string \
            + "_lagorder" + ".txt"
        lagorder_file = os.path.join(here, lagorder_file)
        lagorder_file = open(lagorder_file, encoding='latin_1')
        results["lagorder"] = dict()
        aic_start = "Akaike Info Criterion:"
        fpe_start = "Final Prediction Error:"
        hqic_start = "Hannan-Quinn Criterion:"
        bic_start = "Schwarz Criterion:"
        for line in lagorder_file:
            if line.startswith(aic_start):
                results["lagorder"]["aic"] = int(line[len(aic_start):])
            elif line.startswith(fpe_start):
                results["lagorder"]["fpe"] = int(line[len(fpe_start):])
            elif line.startswith(hqic_start):
                results["lagorder"]["hqic"] = int(line[len(hqic_start):])
            elif line.startswith(bic_start):
                results["lagorder"]["bic"] = int(line[len(bic_start):])
        lagorder_file.close()

        # ---------------------------------------------------------------------
        # parse output related to non-normality-test:
        test_norm_file = dataset.__str__() + "_" + source + "_" + dt_string \
            + "_diag" + ".txt"
        test_norm_file = os.path.join(here, test_norm_file)
        test_norm_file = open(test_norm_file, encoding='latin_1')
        results["test_norm"] = dict()
        section_start_marker = "TESTS FOR NONNORMALITY"
        section_reached = False
        subsection_start_marker = "Introduction to Multiple Time Series A"
        subsection_reached = False
        line_start_statistic = "joint test statistic:"
        line_start_pvalue = " p-value:"
        for line in test_norm_file:
            if not section_reached:
                if section_start_marker in line:
                    section_reached = True  # section w/ relevant results found
                continue
            if not subsection_reached:
                if subsection_start_marker in line:
                    subsection_reached = True
                continue
            if "joint_pvalue" in results["test_norm"].keys():
                break
            if line.startswith(line_start_statistic):
                line_end = line[len(line_start_statistic):]
                results["test_norm"]["joint_test_statistic"] = float(line_end)
            if line.startswith(line_start_pvalue):
                line_end = line[len(line_start_pvalue):]
                results["test_norm"]["joint_pvalue"] = float(line_end)
        test_norm_file.close()

        # ---------------------------------------------------------------------
        # parse output related to testing the whiteness of the residuals:
        whiteness_file = dataset.__str__() + "_" + source + "_" + dt_string \
            + "_diag" + ".txt"
        whiteness_file = os.path.join(here, whiteness_file)
        whiteness_file = open(whiteness_file, encoding='latin_1')
        results["whiteness"] = dict()
        section_start_marker = "PORTMANTEAU TEST"
        order_start = "tested order:"
        statistic_start = "test statistic:"
        p_start = " p-value:"
        adj_statistic_start = "adjusted test statistic:"
        unadjusted_finished = False

        in_section = False
        for line in whiteness_file:
            if not in_section and section_start_marker not in line:
                continue
            if not in_section and section_start_marker in line:
                in_section = True
                continue
            if line.startswith(order_start):
                results["whiteness"]["tested order"] = int(
                        line[len(order_start):])
                continue
            if line.startswith(statistic_start):
                results["whiteness"]["test statistic"] = float(
                        line[len(statistic_start):])
                continue
            if line.startswith(adj_statistic_start):
                results["whiteness"]["test statistic adj."] = float(
                        line[len(adj_statistic_start):])
                continue
            if line.startswith(p_start):  # same for unadjusted and adjusted
                if not unadjusted_finished:
                    results["whiteness"]["p-value"] = \
                        float(line[len(p_start):])
                    unadjusted_finished = True
                else:
                    results["whiteness"]["p-value adjusted"] = \
                        float(line[len(p_start):])
                    break
        whiteness_file.close()

        # ---------------------------------------------------------------------
        if debug_mode:
            print_debug_output(results, dt_string)

        results_dict_per_det_terms[dt_s] = results

    return results_dict_per_det_terms
