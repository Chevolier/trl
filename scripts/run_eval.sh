# checkpoint_nums=(100)
# checkpoint_name="qwen2.5-vl-7bi-20250823-042926"

# for checkpoint_num in "${checkpoint_nums[@]}"; do
#     echo "Processing checkpoint: $checkpoint_num"
#     python evaluation/offline_inference_and_eval.py \
#         --model_path checkpoints/${checkpoint_name}/checkpoint-${checkpoint_num} \
#         --analyse_dir data/group_stand_revised \
#         --gt_file data/group_stand_revised/video_gpt_responses_gt.jsonl \
#         --predict_dir outputs/${checkpoint_name} \
#         --batch_size 16
# done

#!/bin/bash

checkpoint_nums=(100 200 400 500)
checkpoint_names=(
    "qwen2.5-vl-7bi-20250823-042926"
    "qwen2.5-vl-7bi-20250824-231153"
)

# Function to get number of available GPUs
get_gpu_count() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --list-gpus | wc -l
    else
        echo "0"
    fi
}

# Function to run inference on specific GPU
run_checkpoint() {
    local checkpoint_name=$1
    local checkpoint_num=$2
    local gpu_id=$3
    local job_id=$4
    local log_file="logs/${checkpoint_name}_checkpoint_${checkpoint_num}_gpu_${gpu_id}.log"
    
    echo "Job $job_id: Starting $checkpoint_name/checkpoint-$checkpoint_num on GPU $gpu_id"
    
    # Create logs directory if it doesn't exist
    mkdir -p logs
    
    CUDA_VISIBLE_DEVICES=$gpu_id python evaluation/offline_inference_and_eval.py \
        --model_path checkpoints/${checkpoint_name}/checkpoint-${checkpoint_num} \
        --analyse_dir data/group_stand_revised \
        --gt_file data/group_stand_revised/video_gpt_responses_gt.jsonl \
        --predict_dir outputs/${checkpoint_name}/checkpoint-${checkpoint_num} \
        --batch_size 8 \
        > "$log_file" 2>&1
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✅ Job $job_id completed: $checkpoint_name/checkpoint-$checkpoint_num on GPU $gpu_id"
    else
        echo "❌ Job $job_id failed: $checkpoint_name/checkpoint-$checkpoint_num on GPU $gpu_id (exit code: $exit_code)"
    fi
    
    return $exit_code
}

# Check available GPUs
gpu_count=$(get_gpu_count)
echo "Available GPUs: $gpu_count"

if [ $gpu_count -eq 0 ]; then
    echo "Error: No GPUs available!"
    exit 1
fi

# Display GPU information
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits

# Generate job queue
job_queue=()
job_id=0

for checkpoint_name in "${checkpoint_names[@]}"; do
    for checkpoint_num in "${checkpoint_nums[@]}"; do
        job_queue+=("$checkpoint_name|$checkpoint_num|$job_id")
        ((job_id++))
    done
done

total_jobs=${#job_queue[@]}
echo ""
echo "=== JOB CONFIGURATION ==="
echo "Checkpoint names: ${#checkpoint_names[@]}"
for name in "${checkpoint_names[@]}"; do
    echo "  - $name"
done
echo "Checkpoint numbers: ${checkpoint_nums[@]}"
echo "Total jobs: $total_jobs"
echo "Available GPUs: $gpu_count (1 job per GPU at a time)"
echo ""

# Initialize GPU tracking arrays
declare -a gpu_busy           # Track if GPU is busy (0=free, 1=busy)
declare -a gpu_current_pid    # Track current PID on each GPU
declare -a gpu_current_job    # Track current job info on each GPU

for ((i=0; i<gpu_count; i++)); do
    gpu_busy[$i]=0
    gpu_current_pid[$i]=""
    gpu_current_job[$i]=""
done

# Job execution variables
completed_jobs=0
failed_jobs=()
current_job_index=0

echo "=== STARTING JOB EXECUTION ==="
echo "Jobs will be queued and executed one per GPU"
echo ""

# Main execution loop
while [ $completed_jobs -lt $total_jobs ]; do
    
    # Check for completed jobs and free up GPUs
    for ((gpu=0; gpu<gpu_count; gpu++)); do
        if [ ${gpu_busy[$gpu]} -eq 1 ] && [ -n "${gpu_current_pid[$gpu]}" ]; then
            pid=${gpu_current_pid[$gpu]}
            
            # Check if job is still running
            if ! kill -0 $pid 2>/dev/null; then
                # Job completed, get exit code
                wait $pid
                exit_code=$?
                ((completed_jobs++))
                
                # Mark GPU as free
                gpu_busy[$gpu]=0
                job_info=${gpu_current_job[$gpu]}
                
                if [ $exit_code -eq 0 ]; then
                    echo "✅ [$completed_jobs/$total_jobs] Completed: $job_info on GPU $gpu"
                else
                    echo "❌ [$completed_jobs/$total_jobs] Failed: $job_info on GPU $gpu (exit: $exit_code)"
                    failed_jobs+=("$job_info")
                fi
                
                # Clear GPU tracking
                gpu_current_pid[$gpu]=""
                gpu_current_job[$gpu]=""
            fi
        fi
    done
    
    # Start new jobs on free GPUs
    for ((gpu=0; gpu<gpu_count; gpu++)); do
        if [ ${gpu_busy[$gpu]} -eq 0 ] && [ $current_job_index -lt $total_jobs ]; then
            # Get next job from queue
            job=${job_queue[$current_job_index]}
            IFS='|' read -r checkpoint_name checkpoint_num job_id <<< "$job"
            
            # Start job on this GPU
            run_checkpoint "$checkpoint_name" "$checkpoint_num" "$gpu" "$job_id" &
            new_pid=$!
            
            # Mark GPU as busy and track job
            gpu_busy[$gpu]=1
            gpu_current_pid[$gpu]=$new_pid
            gpu_current_job[$gpu]="${checkpoint_name}/checkpoint-${checkpoint_num}"
            
            echo "🚀 Started job $job_id: ${checkpoint_name}/checkpoint-${checkpoint_num} on GPU $gpu (PID: $new_pid)"
            
            ((current_job_index++))
            
            # Small delay to avoid race conditions
            sleep 1
        fi
    done
    
    # Show current status every 30 seconds
    if [ $((SECONDS % 30)) -eq 0 ]; then
        echo ""
        echo "=== STATUS UPDATE ==="
        echo "Completed: $completed_jobs/$total_jobs"
        echo "Queued: $((total_jobs - current_job_index))"
        for ((gpu=0; gpu<gpu_count; gpu++)); do
            if [ ${gpu_busy[$gpu]} -eq 1 ]; then
                echo "GPU $gpu: BUSY - ${gpu_current_job[$gpu]}"
            else
                echo "GPU $gpu: FREE"
            fi
        done
        echo ""
    fi
    
    # Check every 2 seconds
    sleep 2
done

# Final summary
echo ""
echo "=== FINAL SUMMARY ==="
echo "Total jobs: $total_jobs"
echo "Successful jobs: $((total_jobs - ${#failed_jobs[@]}))"
echo "Failed jobs: ${#failed_jobs[@]}"

if [ ${#failed_jobs[@]} -gt 0 ]; then
    echo ""
    echo "Failed jobs:"
    for job in "${failed_jobs[@]}"; do
        echo "  - $job"
    done
    echo ""
    echo "Check log files in ./logs/ directory for details"
    exit 1
else
    echo ""
    echo "🎉 All jobs completed successfully!"
fi

# Show final resource usage
echo ""
echo "=== RESOURCE USAGE ==="
jobs_per_gpu=$((total_jobs / gpu_count))
remaining_jobs=$((total_jobs % gpu_count))

for ((gpu=0; gpu<gpu_count; gpu++)); do
    if [ $gpu -lt $remaining_jobs ]; then
        actual_jobs=$((jobs_per_gpu + 1))
    else
        actual_jobs=$jobs_per_gpu
    fi
    echo "GPU $gpu processed: $actual_jobs jobs"
done

echo ""
echo "All log files saved in ./logs/ directory"
