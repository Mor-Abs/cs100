# This code is cybershake_investigation.gen_emprical_archive

import argparse
from pathlib import Path

from qcore import constants
from cybershake_investigation import empirical_support


def load_args():
    parser = argparse.ArgumentParser(
        description="Script to calculate IMs for empirical models for an archived cybershake version."
        "Produces one .csv for each Realisation containing "
        "all IM data for all sites"
    )
    parser.add_argument(
        "archive_dir",
        type=Path,
        help="The root directory of the archived cybershake version."
        " requires at least the IM and Source folders in each fault",
    )
    parser.add_argument(
        "ll_ffp", type=str, help="The file listing the stations with lat and lon values"
    )
    parser.add_argument(
        "vs30_ffp",
        type=str,
        help="The file listing the stations with their vs30 values",
    )
    parser.add_argument(
        "z_ffp", type=str, help="The file listing stations with their z values"
    )
    parser.add_argument("nhm_ffp", type=Path, help="Path to the nhm file")
    parser.add_argument(
        "meta_config_ffp",
        type=Path,
        help="Path to the meta_config weight file. Found in Empirical util.",
    )
    parser.add_argument(
        "model_config_ffp",
        type=Path,
        help="Path to the model_config file. Found in Empirical util.",
    )
    parser.add_argument("output", type=Path, help="Output Path to generate results")
    parser.add_argument(
        "--ss_db",
        type=str,
        default=None,
        help="The flt_site_source_db path to reduce"
        " the calculation requirements for site source distances",
    )
    parser.add_argument(
        "--component",
        choices=list(constants.Components.iterate_str_values()),
        default=constants.Components.crotd50.str_value,
        help="The component you want to calculate."
        " Available components are: [%(choices)s]. Default is %(default)",
    )
    parser.add_argument(
        "--n_procs",
        type=int,
        default=1,
        help="The number of process to run",
    )

    args = parser.parse_args()
    return args


def main():
    args = load_args()
    empirical_support.calc_empirical_archive(
        args.archive_dir,
        args.ll_ffp,
        args.vs30_ffp,
        args.z_ffp,
        args.nhm_ffp,
        args.meta_config_ffp,
        args.model_config_ffp,
        args.output,
        args.ss_db,
        args.component,
        args.n_procs,
    )


if __name__ == "__main__":
    main()