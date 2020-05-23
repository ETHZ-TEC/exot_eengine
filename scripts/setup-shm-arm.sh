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
