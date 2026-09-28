# Usage: awk -v expected_verify=0|1 -f check-object-log.awk client.log
function fields(   i,a) {
    delete v
    for (i=2; i<=NF; ++i) {
        split($i,a,"=")
        v[a[1]]=a[2]
    }
}
function require(ok, message) {
    if (!ok) {
        print "VALIDATION_FAILURE " message > "/dev/stderr"
        ++failures
    }
}
/^warm up done$/ { warmup=1 }
/^OBJECT_SIZE_PHASE / {
    fields()
    ++phases
    seconds=v["seconds"]+0
    operations=v["operations"]+0
    reads=v["rdma_read_posts"]+0
    read_bytes=v["rdma_read_bytes"]+0
    writes=v["rdma_write_posts"]+0
    write_bytes=v["rdma_write_bytes"]+0
    verify=v["verify"]+0
    require(v["phase"]==1 && v["workers"]==24, "phase/worker shape")
    require(v["object_size"]==512 && v["total_memory_bytes"]==17179869184,
            "original 512-byte/16-GiB workload")
}
/^OBJECT_SIZE_VERIFY_PASS / {
    fields()
    full_verify=(v["objects"]==33554432 && v["bytes_per_object"]==512)
}
/^carbink.compaction_total / {
    fields()
    moved_total=v["moved_spans"]+0
    rollback_total=v["rollback_spans"]+0
    reclaimed_total=v["reclaim_stripes"]+0
    require(v["queue_remaining"]==0, "compaction queue drained")
    require(v["reclaim_bytes"]==reclaimed_total*6*8192, "reclaim byte units")
}
/^carbink.compaction_work / {
    fields()
    moved_work=v["moved_spans"]+0
    rollback_work=v["rollback_spans"]+0
    reclaimed_work=v["reclaim_stripes"]+0
}
/^carbink.pages / {
    fields()
    pages_balanced=(v["created"]==v["released"] && v["created"]>0)
    objects_balanced=(v["objects_created"]==33554432 &&
                      v["objects_released"]==33554432)
}
/^rdma.shutdown_stop / {
    fields()
    shutdown_ok=(v["completed"]==6 && v["failed"]==0 && v["skipped_dead"]==0)
}
/^OBJECT_SIZE_PASS cleanup complete$/ { cleanup=1 }
/^exact used bytes: 0$/ { exact_zero=1 }
/^ERROR:/ { ++runtime_errors }
END {
    require(warmup && phases==1, "full warmup and one phase")
    require(seconds>=20 && operations>0, "nonempty full timing window")
    require(reads>0 && read_bytes==reads*8192, "actual 8-KiB RDMA reads")
    require(writes>0 && write_bytes==writes*8192, "full-span RDMA writes")
    require(verify==expected_verify, "requested verification mode")
    require(!expected_verify || full_verify, "all-object all-byte oracle")
    require(reclaimed_work>0 && moved_work>0, "compaction occurred during work")
    require(pages_balanced && objects_balanced, "page/object lifetime balance")
    require(cleanup && shutdown_ok && exact_zero && !runtime_errors,
            "successful cleanup and shutdown")
    duration=seconds>0 ? seconds : 1
    printf "{\n  \"valid_run\": %s,\n", failures ? "false" : "true"
    printf "  \"full_data_oracle\": %s,\n", full_verify ? "true" : "false"
    printf "  \"phase_seconds\": %.9f,\n  \"operations\": %.0f,\n", seconds,operations
    printf "  \"ops_per_second\": %.9f,\n", operations/duration
    printf "  \"read_posts\": %.0f,\n  \"read_bytes\": %.0f,\n", reads,read_bytes
    printf "  \"read_payload_gbps\": %.9f,\n", read_bytes*8/duration/1e9
    printf "  \"work_reclaimed_groups\": %.0f,\n", reclaimed_work
    printf "  \"work_reclaimed_groups_per_second\": %.9f,\n", reclaimed_work/duration
    printf "  \"work_moved_spans\": %.0f,\n  \"work_rollback_spans\": %.0f,\n",moved_work,rollback_work
    printf "  \"total_reclaimed_groups\": %.0f,\n",reclaimed_total
    printf "  \"total_moved_spans\": %.0f,\n  \"total_rollback_spans\": %.0f\n}\n",moved_total,rollback_total
    exit(failures ? 1 : 0)
}
