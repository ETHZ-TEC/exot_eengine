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
# 
#!/usr/bin/env bash

# By default, if available, we will use the /dev/hugepages pool/mount. If it is
# not available, we will set up a separate mountpoint with a hugetlbfs mount
# added to /etc/fstab.
#
# The configuration parameters are:
#   - SHM_MOUNTPOINT : directory where the hugetlbfs will be mounted
#   - SHM_GROUP      : group which will have the hugepages support enabled for
#   - USERS          : users to add to the SHM group
#   - SIZES          : list of sizes (in MiB) of huge page shm files to create
SHM_MOUNTPOINT="/shm/"
SHM_GROUP="shm"
USERS=( "exot" )
SIZES=$(seq 2 2 8)

u:info() { (>&2 echo -e "[\e[1m\e[34minfo\e[0m]\t" $@;); }
u:warn() { (>&2 echo -e "[\e[1m\e[33mwarn\e[0m]\t" $@;); }
u:erro() { (>&2 echo -e "[\e[1m\e[31merror\e[0m]\t" $@;); exit 1; }

# set sysctl kernel hugepages
grep -q "nr_hugepages"      /etc/sysctl.conf || {
    var="vm.nr_hugepages = 8"
    echo "$var" | sudo tee -a /etc/sysctl.conf      \
        && u:info "added \"$var\" to sysctl.conf"   \
        || u:erro "failed to add \"$var\" to sysctl.conf"
}

# create huge pages shared memory files
test -d /dev/hugepages && {
    for size in $SIZES; do \
        mnt="/dev/hugepages"
        file="$mnt/$size"
        msg=`printf "%-4s MiB huge pages file @ %s\n" "$size" "$file"`
        sudo truncate -s $((1024 * 1024 * size)) $file  \
            && u:info "created $msg"                    \
            || u:warn "failed to create $msg"
    done
}
