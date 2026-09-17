# TabFM Codebase Part 1

## File: .gitignore

```
# Compiled python modules.
*.pyc

# Byte-compiled
_pycache__/
.cache/

# Poetry, setuptools, PyPI distribution artifacts.
/*.egg-info
.eggs/
build/
dist/
poetry.lock

# Tests
.pytest_cache/

# Type checking
.pytype/

# Other
*.DS_Store

# PyCharm
.idea

# Ignore generated docs
docs/_build
docs/api
.venv/


```

## File: .pylintrc

```
# This Pylint rcfile contains a best-effort configuration to uphold the
# best-practices and style described in the Google Python style guide:
#   https://google.github.io/styleguide/pyguide.html
#
# Its canonical open-source location is:
#   https://google.github.io/styleguide/pylintrc

[MAIN]

# Files or directories to be skipped. They should be base names, not paths.
ignore=third_party

# Files or directories matching the regex patterns are skipped. The regex
# matches against base names, not paths.
ignore-patterns=

# Pickle collected data for later comparisons.
persistent=no

# List of plugins (as comma separated values of python modules names) to load,
# usually to register additional checkers.
load-plugins=

# Use multiple processes to speed up Pylint.
jobs=4

# Allow loading of arbitrary C extensions. Extensions are imported into the
# active Python interpreter and may run arbitrary code.
unsafe-load-any-extension=no


[MESSAGES CONTROL]

# Only show warnings with the listed confidence levels. Leave empty to show
# all. Valid levels: HIGH, INFERENCE, INFERENCE_FAILURE, UNDEFINED
confidence=

# Enable the message, report, category or checker with the given id(s). You can
# either give multiple identifier separated by comma (,) or put this option
# multiple time (only on the command line, not in the configuration file where
# it should appear only once). See also the "--disable" option for examples.
#enable=

# Disable the message, report, category or checker with the given id(s). You
# can either give multiple identifiers separated by comma (,) or put this
# option multiple times (only on the command line, not in the configuration
# file where it should appear only once).You can also use "--disable=all" to
# disable everything first and then reenable specific checks. For example, if
# you want to run only the similarities checker, you can use "--disable=all
# --enable=similarities". If you want to run only the classes checker, but have
# no Warning level messages displayed, use"--disable=all --enable=classes
# --disable=W"
disable=R,
        abstract-method,
        apply-builtin,
        arguments-differ,
        attribute-defined-outside-init,
        backtick,
        bad-option-value,
        basestring-builtin,
        buffer-builtin,
        c-extension-no-member,
        consider-using-enumerate,
        cmp-builtin,
        cmp-method,
        coerce-builtin,
        coerce-method,
        delslice-method,
        div-method,
        eq-without-hash,
        execfile-builtin,
        file-builtin,
        filter-builtin-not-iterating,
        fixme,
        getslice-method,
        global-statement,
        hex-method,
        idiv-method,
        implicit-str-concat,
        import-error,
        import-self,
        import-star-module-level,
        input-builtin,
        intern-builtin,
        invalid-str-codec,
        locally-disabled,
        long-builtin,
        long-suffix,
        map-builtin-not-iterating,
        misplaced-comparison-constant,
        missing-function-docstring,
        metaclass-assignment,
        next-method-called,
        next-method-defined,
        no-absolute-import,
        no-init,  # added
        no-member,
        no-name-in-module,
        no-self-use,
        nonzero-method,
        oct-method,
        old-division,
        old-ne-operator,
        old-octal-literal,
        old-raise-syntax,
        parameter-unpacking,
        print-statement,
        raising-string,
        range-builtin-not-iterating,
        raw_input-builtin,
        rdiv-method,
        reduce-builtin,
        relative-import,
        reload-builtin,
        round-builtin,
        setslice-method,
        signature-differs,
        standarderror-builtin,
        suppressed-message,
        sys-max-int,
        trailing-newlines,
        unichr-builtin,
        unicode-builtin,
        unnecessary-pass,
        unpacking-in-except,
        useless-else-on-loop,
        useless-suppression,
        using-cmp-argument,
        wrong-import-order,
        xrange-builtin,
        zip-builtin-not-iterating,


[REPORTS]

# Set the output format. Available formats are text, parseable, colorized, msvs
# (visual studio) and html. You can also give a reporter class, eg
# mypackage.mymodule.MyReporterClass.
output-format=text

# Tells whether to display a full report or only the messages
reports=no

# Python expression which should return a note less than 10 (10 is the highest
# note). You have access to the variables errors warning, statement which
# respectively contain the number of errors / warnings messages and the total
# number of statements analyzed. This is used by the global evaluation report
# (RP0004).
evaluation=10.0 - ((float(5 * error + warning + refactor + convention) / statement) * 10)

# Template used to display messages. This is a python new-style format string
# used to format the message information. See doc for all details
#msg-template=


[BASIC]

# Good variable names which should always be accepted, separated by a comma
good-names=main,_

# Bad variable names which should always be refused, separated by a comma
bad-names=

# Colon-delimited sets of names that determine each other's naming style when
# the name regexes allow several styles.
name-group=

# Include a hint for the correct naming format with invalid-name
include-naming-hint=no

# List of decorators that produce properties, such as abc.abstractproperty. Add
# to this list to register other decorators that produce valid properties.
property-classes=abc.abstractproperty,cached_property.cached_property,cached_property.threaded_cached_property,cached_property.cached_property_with_ttl,cached_property.threaded_cached_property_with_ttl

# Regular expression matching correct function names
function-rgx=^(?:(?P<exempt>setUp|tearDown|setUpModule|tearDownModule)|(?P<camel_case>_?[A-Z][a-zA-Z0-9]*)|(?P<snake_case>_?[a-z][a-z0-9_]*))$

# Regular expression matching correct variable names
variable-rgx=^[a-z][a-z0-9_]*$

# Regular expression matching correct constant names
const-rgx=^(_?[A-Z][A-Z0-9_]*|__[a-z0-9_]+__|_?[a-z][a-z0-9_]*)$

# Regular expression matching correct attribute names
attr-rgx=^_{0,2}[a-z][a-z0-9_]*$

# Regular expression matching correct argument names
argument-rgx=^[a-z][a-z0-9_]*$

# Regular expression matching correct class attribute names
class-attribute-rgx=^(_?[A-Z][A-Z0-9_]*|__[a-z0-9_]+__|_?[a-z][a-z0-9_]*)$

# Regular expression matching correct inline iteration names
inlinevar-rgx=^[a-z][a-z0-9_]*$

# Regular expression matching correct class names
class-rgx=^_?[A-Z][a-zA-Z0-9]*$

# Regular expression matching correct module names
module-rgx=^(_?[a-z][a-z0-9_]*|__init__)$

# Regular expression matching correct method names
method-rgx=(?x)^(?:(?P<exempt>_[a-z0-9_]+__|runTest|setUp|tearDown|setUpTestCase|tearDownTestCase|setupSelf|tearDownClass|setUpClass|(test|assert)_*[A-Z0-9][a-zA-Z0-9_]*|next)|(?P<camel_case>_{0,2}[A-Z][a-zA-Z0-9_]*)|(?P<snake_case>_{0,2}[a-z][a-z0-9_]*))$

# Regular expression which should only match function or class names that do
# not require a docstring.
no-docstring-rgx=(__.*__|main|test.*|.*test|.*Test)$

# Minimum line length for functions/classes that require docstrings, shorter
# ones are exempt.
docstring-min-length=12


[TYPECHECK]

# List of decorators that produce context managers, such as
# contextlib.contextmanager. Add to this list to register other decorators that
# produce valid context managers.
contextmanager-decorators=contextlib.contextmanager,contextlib2.contextmanager

# List of module names for which member attributes should not be checked
# (useful for modules/projects where namespaces are manipulated during runtime
# and thus existing member attributes cannot be deduced by static analysis. It
# supports qualified module names, as well as Unix pattern matching.
ignored-modules=

# List of class names for which member attributes should not be checked (useful
# for classes with dynamically set attributes). This supports the use of
# qualified names.
ignored-classes=optparse.Values,thread._local,_thread._local

# List of members which are set dynamically and missed by pylint inference
# system, and so shouldn't trigger E1101 when accessed. Python regular
# expressions are accepted.
generated-members=


[FORMAT]

# Maximum number of characters on a single line.
max-line-length=80

# TODO(https://github.com/pylint-dev/pylint/issues/3352): Direct pylint to exempt
# lines made too long by directives to pytype.

# Regexp for a line that is allowed to be longer than the limit.
ignore-long-lines=(?x)(
  ^\s*(\#\ )?<?https?://\S+>?$|
  ^\s*(from\s+\S+\s+)?import\s+.+$)

# Allow the body of an if to be on the same line as the test if there is no
# else.
single-line-if-stmt=yes

# Maximum number of lines in a module
max-module-lines=99999

# String used as indentation unit.  The internal Google style guide mandates 2
# spaces.  Google's externaly-published style guide says 4, consistent with
# PEP 8.  Here, we use 2 spaces, for conformity with many open-sourced Google
# projects (like TensorFlow).
indent-string='  '

# Number of spaces of indent required inside a hanging  or continued line.
indent-after-paren=4

# Expected format of line ending, e.g. empty (any line ending), LF or CRLF.
expected-line-ending-format=


[MISCELLANEOUS]

# List of note tags to take in consideration, separated by a comma.
notes=TODO


[STRING]

# This flag controls whether inconsistent-quotes generates a warning when the
# character used as a quote delimiter is used inconsistently within a module.
check-quote-consistency=yes


[VARIABLES]

# Tells whether we should check for unused import in __init__ files.
init-import=no

# A regular expression matching the name of dummy variables (i.e. expectedly
# not used).
dummy-variables-rgx=^\*{0,2}(_$|unused_|dummy_)

# List of additional names supposed to be defined in builtins. Remember that
# you should avoid to define new builtins when possible.
additional-builtins=

# List of strings which can identify a callback function by name. A callback
# name must start or end with one of those strings.
callbacks=cb_,_cb

# List of qualified module names which can have objects that can redefine
# builtins.
redefining-builtins-modules=six,six.moves,past.builtins,future.builtins,functools


[LOGGING]

# Logging modules to check that the string format arguments are in logging
# function parameter format
logging-modules=logging,absl.logging,tensorflow.io.logging


[SIMILARITIES]

# Minimum lines number of a similarity.
min-similarity-lines=4

# Ignore comments when computing similarities.
ignore-comments=yes

# Ignore docstrings when computing similarities.
ignore-docstrings=yes

# Ignore imports when computing similarities.
ignore-imports=no


[SPELLING]

# Spelling dictionary name. Available dictionaries: none. To make it working
# install python-enchant package.
spelling-dict=

# List of comma separated words that should not be checked.
spelling-ignore-words=

# A path to a file that contains private dictionary; one word per line.
spelling-private-dict-file=

# Tells whether to store unknown words to indicated private dictionary in
# --spelling-private-dict-file option instead of raising a message.
spelling-store-unknown-words=no


[IMPORTS]

# Deprecated modules which should not be used, separated by a comma
deprecated-modules=regsub,
                   TERMIOS,
                   Bastion,
                   rexec,
                   sets

# Create a graph of every (i.e. internal and external) dependencies in the
# given file (report RP0402 must not be disabled)
import-graph=

# Create a graph of external dependencies in the given file (report RP0402 must
# not be disabled)
ext-import-graph=

# Create a graph of internal dependencies in the given file (report RP0402 must
# not be disabled)
int-import-graph=

# Force import order to recognize a module as part of the standard
# compatibility libraries.
known-standard-library=

# Force import order to recognize a module as part of a third party library.
known-third-party=enchant, absl

# Analyse import fallback blocks. This can be used to support both Python 2 and
# 3 compatible code, which means that the block might have code that exists
# only in one or another interpreter, leading to false positives when analysed.
analyse-fallback-blocks=no


[CLASSES]

# List of method names used to declare (i.e. assign) instance attributes.
defining-attr-methods=__init__,
                      __new__,
                      setUp

# List of member names, which should be excluded from the protected access
# warning.
exclude-protected=_asdict,
                  _fields,
                  _replace,
                  _source,
                  _make

# List of valid names for the first argument in a class method.
valid-classmethod-first-arg=cls,
                            class_

# List of valid names for the first argument in a metaclass class method.
valid-metaclass-classmethod-first-arg=mcs

```

## File: BUILD

```
# Bazel workspace root package declaration.
# This file makes the root package visible to Bazel.

```

## File: CHANGELOG.md

```
# Changelog

<!--

Changelog follow the https://keepachangelog.com/ standard (at least the headers)

This allow to:

* auto-parsing release notes during the automated releases from github-action:
  https://github.com/marketplace/actions/pypi-github-auto-release
* Have clickable headers in the rendered markdown

To release a new version (e.g. from `1.0.0` -> `2.0.0`):

* Create a new `# [2.0.0] - YYYY-MM-DD` header and add the current
  `[Unreleased]` notes.
* At the end of the file:
  * Define the new link url:
  `[2.0.0]: https://github.com/google-research/tabfm/compare/v1.0.0...v2.0.0`
  * Update the `[Unreleased]` url: `v1.0.0...HEAD` -> `v2.0.0...HEAD`

-->

## [1.0.1] - 2026-07-09

### Fixed

* PyTorch weight loading: load `model.safetensors` via `PyTorchModelHubMixin`.
  The `1.0.0` loader looked for `pytorch_model.bin`, which the Hugging Face
  checkpoint no longer provides, so `load()` raised `FileNotFoundError`.
* `EnsembleGenerator` no longer re-transforms the full training set on every
  `predict` call (prediction cost now scales with query size, not context size).
* Query-axis mask/bias collapse in the memory-efficient (FLASH) JAX attention.
* `predict` on multi-device hosts no longer crashes (IndivisibleError / device
  mismatch).
* `TabFMRegressor.predict` before `fit` now raises `NotFittedError`.
* `TabFMClassifier.predict` no longer returns object-dtype labels.
* README regression example now loads the regression checkpoint.
* Loading a checkpoint whose type does not match the estimator now fails fast
  with an actionable error, instead of a cryptic squeeze error (classification
  weights in `TabFMRegressor`) or silently wrong predictions (regression
  weights in `TabFMClassifier`).
* The sklearn layer handles duplicate and non-string column names: duplicates
  fail fast with a clear message, and datetime columns with integer labels no
  longer crash.
* Fitted estimators and the PyTorch model are now picklable, which AutoGluon /
  TabArena require to save a fitted model. The PyTorch gelu activation is a
  module-level function, and the JAX estimators drop their memoized compiled
  predict functions on pickle and rebuild them lazily.

### Changed

* PyTorch model runs in bfloat16 by default, matching the JAX compute dtype.
* Activation chunking is enabled by default to bound peak memory on large tasks.
* JAX and PyTorch models gained Hugging Face Hub support (`from_pretrained` /
  `save_pretrained`); weight downloads are narrowed to the requested model type.

## [1.0.0] - 2026-06-29

* Initial release

[1.0.1]: https://github.com/google-research/tabfm/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/google-research/tabfm/releases/tag/v1.0.0

```

## File: CONTRIBUTING.md

```
# How to Contribute

We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement (CLA). You (or your employer) retain the copyright to your
contribution; this simply gives us permission to use and redistribute your
contributions as part of the project. Head over to
<https://cla.developers.google.com/> to see your current agreements on file or
to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.

## Code Reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

## Community Guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).

```

## File: LICENSE

```

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

## File: MODULE.bazel

```
module(
    name = "tabfm",
    version = "1.0.0",
)

bazel_dep(name = "rules_python", version = "0.26.0")

pip = use_extension("@rules_python//python/extensions:pip.bzl", "pip")

pip.parse(
    hub_name = "pip",
    python_version = "3.11",
    requirements_lock = "//:requirements.txt",
    extra_pip_args = ["--index-url", "https://pypi.org/simple/", "--extra-index-url", "https://download.pytorch.org/whl/cpu"],
)

use_repo(pip, "pip")

```

## File: README.md

```
# TabFM: Tabular Foundation Models

TabFM (Tabular Foundation Model) is a scikit-learn compatible tabular foundation model. It allows you to perform zero-shot classification and regression on tabular datasets with mixed column types out-of-the-box.

At inference time, TabFM does not require training parameters on your dataset; instead, it leverages in-context learning by reading your training data as "context" to make instant predictions on new test samples.

*This is not an officially supported Google product.*

---

## Installation

To install TabFM, clone the repository and install it locally with the backend of your choice:

**JAX (CPU):**
```bash
git clone https://github.com/google-research/tabfm.git
cd tabfm
pip install -e .[jax]
```

**JAX (GPU):**
```bash
git clone https://github.com/google-research/tabfm.git
cd tabfm
pip install -e .[jax,cuda]
```

**PyTorch (CPU/GPU):**
```bash
git clone https://github.com/google-research/tabfm.git
cd tabfm
pip install -e .[pytorch]
```
*Note: For PyTorch with GPU support, ensure you have the appropriate PyTorch version installed for your CUDA version before installing TabFM.*

### Requirements
For a complete list of pinned dependencies and versions, please see [requirements.txt](requirements.txt). The core requirements depend on the backend you choose:
*   Python >= 3.11
*   Hugging Face Hub (for downloading pre-trained weights)
*   **JAX Backend:**
    *   JAX (specifically `jax==0.10.1`)
    *   Flax (specifically `flax==0.12.7`, using the modern `flax.nnx` API)
*   **PyTorch Backend:**
    *   PyTorch (specifically `torch==2.12.1+cpu` or a GPU version)

---

## License notice for pretrained weights

> **Important:** The TabFM source code in this repository is licensed under
> Apache-2.0. However, the default Quick Start calls `tabfm_v1_0_0.load()`,
> which automatically downloads pretrained weights from Hugging Face. Those
> pretrained weights are distributed under the separate
> `tabfm-non-commercial-v1.0` license and are restricted to non-commercial,
> non-production use. Commercial or production use of the default pretrained
> weights is **not permitted**.

---

## Quick Start (TabFM v1.0.0)

We provide pre-trained weights for the **TabFM v1.0.0** release. The library handles downloading and loading these weights automatically from Hugging Face. These default weights are governed by the separate `tabfm-non-commercial-v1.0` license described above.
### 1. Classification Example

```python
import numpy as np
import pandas as pd
from tabfm import TabFMClassifier

# Choose your backend:

# OPTION A: JAX Backend
from tabfm import tabfm_v1_0_0_jax as tabfm_v1_0_0
model = tabfm_v1_0_0.load()

# OPTION B: PyTorch Backend
# from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
# model = tabfm_v1_0_0.load()

# Initialize scikit-learn compatible classifier (works with either backend model)
clf = TabFMClassifier(model=model)

# Prepare your dataset (supports mixed numerical and categorical features)
X_train = pd.DataFrame({
    "age": [25.0, 45.0, 35.0, 50.0],
    "job": ["engineer", "manager", "engineer", "manager"],
    "income": [80000, 120000, 90000, 130000]
})
y_train = np.array(["low_risk", "high_risk", "low_risk", "high_risk"])

X_test = pd.DataFrame({
    "age": [30.0, 48.0],
    "job": ["engineer", "manager"],
    "income": [85000, 125000]
})

# Fit classifier (prepares ordinal encoders and numerical scalers)
clf.fit(X_train, y_train)

# Predict classes and probabilities
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)

print("Predictions:", predictions)
print("Class Probabilities:\n", probabilities)
```

### 2. Regression Example

```python
import numpy as np
import pandas as pd
from tabfm import TabFMRegressor

# Choose your backend:

# OPTION A: JAX Backend
from tabfm import tabfm_v1_0_0_jax as tabfm_v1_0_0
model = tabfm_v1_0_0.load(model_type="regression")

# OPTION B: PyTorch Backend
# from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
# model = tabfm_v1_0_0.load(model_type="regression")

# Initialize scikit-learn compatible regressor (works with either backend model)
reg = TabFMRegressor(model=model)

# Prepare your dataset
X_train = pd.DataFrame({
    "sqft": [1200, 2500, 1500, 3000],
    "neighborhood": ["A", "B", "A", "C"]
})
y_train = np.array([250000, 550000, 310000, 620000])

X_test = pd.DataFrame({
    "sqft": [1800, 2800],
    "neighborhood": ["A", "B"]
})

# Fit and Predict
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)

print("Predicted Prices:", predictions)
```

---

## Examples Directory

You can find runnable scripts for both classification and regression under the [examples/](examples/) folder:
*   [classification_example.py](examples/classification_example.py)
*   [regression_example.py](examples/regression_example.py)

To run them, simply execute:
```bash
python examples/classification_example.py
```
*(You can edit these files to switch between JAX and PyTorch backends as shown in the comments inside them).*

---

## Evaluation Results

Our model evaluation results can be found in [results/](results/).

---

## FAQ

### Is there a maximum table size for TabFM inputs?

TabFM uses in-context learning over a bounded context window, so very large
tables should be sampled or split before inference. The scikit-learn estimators
expose the main practical limits through `max_num_features` and `max_num_rows`
(defaults are 500 features and 100 context rows), plus `n_estimators` for
ensembling over multiple sampled contexts and `inference_batch_size` for memory
control. If your dataset is larger than these limits, TabFM will work with the
sampled/context rows rather than consuming the full table at once.

### Is there a TabFM technical report or paper?

A technical report is not included in this repository at this time. If a report
or paper describing the architecture, training pipeline, datasets, and evaluation
methodology is released, this README will be updated with a link.

---
## Running Tests

You can run the unit tests directly using Python's `unittest` module:

```bash
# Run all tests (requires both JAX and PyTorch installed)
PYTHONPATH=. python3 -m unittest discover -s tabfm/src/ -p "*_test.py"

# Or run specific test files:
PYTHONPATH=. python3 -m unittest tabfm/src/pytorch/model_test.py
PYTHONPATH=. python3 -m unittest tabfm/src/classifier_and_regressor_pytorch_test.py
```

Alternatively, if you have Bazel installed, you can run tests with:

```bash
bazel test //...
```

```

## File: WORKSPACE

```
workspace(name = "tabfm")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Rules Python
http_archive(
    name = "rules_python",
    sha256 = "9d04041ac92a0985e344235f5d946f71ac543f1b1565f2cdbc9a2aaee8adf55b",
    strip_prefix = "rules_python-0.26.0",
    url = "https://github.com/bazelbuild/rules_python/archive/0.26.0.tar.gz",
)

load("@rules_python//python:repositories.bzl", "py_repositories")
py_repositories()

load("@rules_python//python:pip.bzl", "pip_parse")

# Parse pip dependencies from pyproject.toml
pip_parse(
    name = "pip",
    requirements_lock = "//:pyproject.toml",
)

load("@pip//:requirements.bzl", "install_deps")
install_deps()

```

## File: conftest.py

```
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pytest configuration.

The test suite uses ``absltest.TestCase`` helpers (e.g. ``create_tempdir``),
which read absl flags such as ``--test_tmpdir``. Those flags are only parsed by
``absltest.main()``; under the pytest runner they are never parsed, raising
``UnparsedFlagAccessError``. Parse them here so absltest-based tests work under
pytest.
"""

import importlib.util
import sys

from absl import flags

# Backend-specific test modules import optional extras at import time (torch for
# the pytorch backend; jax/flax/chex for the jax backend). The CI "core tests"
# job runs `pip install -e .[dev]`, which pulls neither extra, so these modules
# would otherwise fail collection with ModuleNotFoundError. Skip them when their
# backend is not installed.
_has_torch = importlib.util.find_spec("torch") is not None
_has_jax = importlib.util.find_spec("jax") is not None
collect_ignore = []
if not _has_torch:
  collect_ignore.append("tabfm/src/classifier_and_regressor_pytorch_test.py")
if not _has_jax:
  collect_ignore += [
      "tabfm/src/jax/model_test.py",
      "tabfm/src/jax/checkpointing_test.py",
      "tabfm/src/jax/memory_efficient_attention_test.py",
  ]
# pytorch/model_test.py is a torch<->jax parity test: it imports both flax and
# torch, so it needs *both* backends installed.
if not (_has_torch and _has_jax):
  collect_ignore.append("tabfm/src/pytorch/model_test.py")


def pytest_configure(config):  # noqa: D401  (pytest hook)
  del config  # Unused.
  if not flags.FLAGS.is_parsed():
    flags.FLAGS(sys.argv[:1])

```

