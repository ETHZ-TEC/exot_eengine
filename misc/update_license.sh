# Copyright (c) 2015-2020, Swiss Federal Institute of Technology (ETH Zurich)
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# 
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# TODO These parameters could also be determined by examining the files
NUM_LINES_COPYRIGHT_OLD=28
NUM_LINES_COPYRIGHT_NEW=28

for searchpattern in "*.m" "*.py" "*.ipynb" "*.bash" "*.sh" "*.bat" "*.toml" "*.cmake" "CMakeLists.txt" "*.h" "*.cpp" "*.cc" "*.java" "*.gradle"; do
  if [ "$searchpattern" == "test.py" ]; then 
    COMMENT="#"
  elif [ "$searchpattern" == "*.m" ]; then
    COMMENT="%"
  elif [ "$searchpattern" == "*.py" ]; then 
    COMMENT="#"
  elif [ "$searchpattern" == "*.ipynb" ]; then
    COMMENT=""
  elif [ "$searchpattern" == "*.bash" ]; then
    COMMENT="#"
  elif [ "$searchpattern" == "*.sh" ]; then
    COMMENT="#"
  elif [ "$searchpattern" == "*.bat" ]; then
    COMMENT="#"
  elif [ "$searchpattern" == "*.toml" ]; then
    COMMENT="#"
  elif [ "$searchpattern" == "*.cmake" ]; then
    COMMENT="#"
  elif [ "$searchpattern" == "CMakeLists.txt" ]; then
    COMMENT="#"
  elif [ "$searchpattern" == "*.h" ]; then
    COMMENT="\/\/"
  elif [ "$searchpattern" == "*.cpp" ]; then
    COMMENT="\/\/"
  elif [ "$searchpattern" == "*.cc" ]; then
    COMMENT="\/\/"
  elif [ "$searchpattern" == "*.java" ]; then
    COMMENT="\/\/"
  elif [ "$searchpattern" == "*.gradle" ]; then
    COMMENT="\/\/"
  fi
  for repo in "app_apk" "app_lib" "app_unx" "datapro" "compilation"; do
    for file in $(find ../../${repo} -name ${searchpattern} -not -path "."                          \
                                                            -not -path "*/data/*"                   \
                                                            -not -path "*.local*"                   \
                                                            -not -path "*/vendor/*"                 \
                                                            -not -path "*/tools/docker/*"           \
                                                            -not -path "*/libnative/src/exot-c++/*" \
                                                            -not -path "*/app_unx/lib/*"            \
                                                            -not -path "*/.cxx/*"                   \
                                                            -not -path "*/build/*"                  \
                                                            ); do 
      if [ "$searchpattern" == "*.ipynb" ]; then
        if ! grep -q "Copyright (c) 2015-" ${file}; then
          echo "Add copyright to ${file}"
          sed -i -e "1,2d" ${file}
        else
          sed -i -e "1,$((${NUM_LINES_COPYRIGHT_OLD} + 8))d" ${file}
          echo "Update copyright of ${file}"
        fi
        echo '{'                            >> ${file}.tmp
        echo ' "cells": ['                  >> ${file}.tmp
        echo '  {'                          >> ${file}.tmp
        echo '   "cell_type": "markdown",'  >> ${file}.tmp
        echo '   "metadata": {},'           >> ${file}.tmp
        echo '   "source": ['               >> ${file}.tmp
        sed -e "s/\"/\\\\\"/g" ../../${repo}/LICENSE >> ${file}.tmp
        echo ' ]'                           >> ${file}.tmp
        echo '},'                           >> ${file}.tmp
        cat ${file}                         >> ${file}.tmp
        mv ${file}.tmp ${file}
        sed -i -e 7"s/.*/    \" &\\\\n\\\\n\", /" ${file}
        sed -i -e 8,$((${NUM_LINES_COPYRIGHT_NEW} + 5))"s/.*/    \" &\\\\n\", /" ${file}
        sed -i -e $((${NUM_LINES_COPYRIGHT_NEW} + 6))"s/.*/    \" &\"/" ${file}
      elif [ -f "${file}" ]; then
        if ! grep -q "Copyright (c) 2015-" ${file}; then
          echo "Add copyright to ${file}"
        else
          sed -i -e "1,${NUM_LINES_COPYRIGHT_OLD}d" ${file}
          echo "Update copyright of ${file}"
        fi
        cat ../../${repo}/LICENSE ${file} > ${file}.tmp && mv ${file}.tmp ${file}
        sed -i -e 1,${NUM_LINES_COPYRIGHT_NEW}"s/.*/${COMMENT} &/" ${file}
      fi
    done
  done
done

