#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <cstdio>
#include <csignal>
#include <cuda_runtime.h>

#define NUM_INPUTS 9
#define NUM_OUTPUTS 9
#define MAX_NODES 1200
#define INITIAL_NODES 80
#define GROWTH_STEP 20
#define GENERATIONS 100000000
#define LAMBDA 256 

#define CHECKPOINT_FILE "cgp_cuda_checkpoint.bin"
#define CHECKPOINT_INTERVAL 10000

using namespace std;

const char* FUNC_NAMES[] = {"BUF", "NOT", "AND", "OR", "XOR", "NAND", "NOR", "XNOR", "MUX"};

// Global flag to safely catch Ctrl+C
volatile sig_atomic_t keep_running = 1;
void int_handler(int dummy) {
    keep_running = 0;
}

// =====================================================
// Data Structures
// =====================================================
struct Node {
    int in1, in2, in3, op;
};

struct Chromosome {
    Node nodes[MAX_NODES];
    int outputs[NUM_OUTPUTS];
};

struct Fitness {
    double weighted;
    int raw_total;
    int highest_failing;

    bool operator<(const Fitness& other) const {
        return weighted < other.weighted;
    }
    bool operator<=(const Fitness& other) const {
        return weighted <= other.weighted;
    }
};

// =====================================================
// Binary Checkpoint I/O
// =====================================================
void save_checkpoint(const char* filename, int gen, const Chromosome& chrom, const Fitness& fit, int last_imp_gen, int current_nodes) {
    FILE* f = fopen(filename, "wb");
    if (!f) return;
    fwrite(&gen, sizeof(int), 1, f);
    fwrite(&chrom, sizeof(Chromosome), 1, f);
    fwrite(&fit, sizeof(Fitness), 1, f);
    fwrite(&last_imp_gen, sizeof(int), 1, f);
    fwrite(&current_nodes, sizeof(int), 1, f);
    fclose(f);
}

bool load_checkpoint(const char* filename, int& gen, Chromosome& chrom, Fitness& fit, int& last_imp_gen, int& current_nodes) {
    FILE* f = fopen(filename, "rb");
    if (!f) return false;

    // Get total file size
    fseek(f, 0, SEEK_END);
    long filesize = ftell(f);
    fseek(f, 0, SEEK_SET);

    // The metadata at the end of the file is exactly 24 bytes (fit(16) + last_imp(4) + curr_nodes(4))
    // The generation at the start is 4 bytes.
    // Therefore, the raw Chromosome chunk in the middle is exactly (filesize - 28) bytes.
    long chrom_size = filesize - 28;
    if (chrom_size <= 0) { fclose(f); return false; }

    if (fread(&gen, sizeof(int), 1, f) != 1) { fclose(f); return false; }

    // Read the entire Chromosome block as raw bytes (ignoring C++ struct padding)
    char* chrom_buffer = new char[chrom_size];
    if (fread(chrom_buffer, 1, chrom_size, f) != chrom_size) {
        delete[] chrom_buffer; fclose(f); return false;
    }

    // Read the trailing metadata
    if (fread(&fit, sizeof(Fitness), 1, f) != 1) { delete[] chrom_buffer; fclose(f); return false; }
    if (fread(&last_imp_gen, sizeof(int), 1, f) != 1) { delete[] chrom_buffer; fclose(f); return false; }
    if (fread(&current_nodes, sizeof(int), 1, f) != 1) { delete[] chrom_buffer; fclose(f); return false; }
    fclose(f);

    // --- UNPACK RAW BYTES ---
    // Calculate how many nodes were actually saved. 
    // Integer division safely discards any compiler padding bytes.
    int saved_max_nodes = (chrom_size - (NUM_OUTPUTS * sizeof(int))) / sizeof(Node);
    
    if (saved_max_nodes != MAX_NODES) {
        cout << "\n[*] Format Shift: Adapting checkpoint from " << saved_max_nodes << " to " << MAX_NODES << " nodes..." << endl;
    }

    int nodes_to_copy = min((int)saved_max_nodes, (int)MAX_NODES);
    
    // Unpack Nodes
    memcpy(chrom.nodes, chrom_buffer, nodes_to_copy * sizeof(Node));
    // Unpack Outputs (located exactly after the saved nodes array)
    memcpy(chrom.outputs, chrom_buffer + (saved_max_nodes * sizeof(Node)), NUM_OUTPUTS * sizeof(int));
    
    delete[] chrom_buffer;

    // Fill new capacity with valid sandbox logic if up-sizing (e.g., 1200 -> 2000)
    if (saved_max_nodes < MAX_NODES) {
        for(int i = saved_max_nodes; i < MAX_NODES; i++) {
            int max_source = NUM_INPUTS + i - 1;
            chrom.nodes[i] = { rand() % (max_source+1), rand() % (max_source+1), rand() % (max_source+1), rand() % 9 };
        }
    }

    // Safety Catch: Repair stagnation counter bug if present
    if (gen == 100000000) gen = last_imp_gen;

    return true;
}

// =====================================================
// CUDA Device Kernel (Ultra-Fast Bit-Sliced Evaluator)
// =====================================================
__device__ uint64_t execute_gate(int op, uint64_t a, uint64_t b, uint64_t c) {
    switch (op) {
        case 0: return a;
        case 1: return ~a;
        case 2: return a & b;
        case 3: return a | b;
        case 4: return a ^ b;
        case 5: return ~(a & b);
        case 6: return ~(a | b);
        case 7: return ~(a ^ b);
        case 8: return (a & b) | ((~a) & c);
        default: return a;
    }
}

__global__ void evaluate_population(
    Chromosome* children,
    uint64_t* d_inputs,   
    uint64_t* d_targets,  
    Fitness* d_fitnesses,
    int current_nodes
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= LAMBDA) return;

    Chromosome chrom = children[tid];
    int errors_per_bit[NUM_OUTPUTS] = {0};

    for (int w = 0; w < 4; w++) {
        uint64_t state[NUM_INPUTS + MAX_NODES];

        for (int i = 0; i < NUM_INPUTS; i++) {
            state[i] = d_inputs[i * 4 + w];
        }

        for (int i = 0; i < current_nodes; i++) {
            Node n = chrom.nodes[i];
            state[NUM_INPUTS + i] = execute_gate(n.op, state[n.in1], state[n.in2], state[n.in3]);
        }

        for (int o = 0; o < NUM_OUTPUTS; o++) {
            uint64_t out_val = state[chrom.outputs[o]];
            uint64_t target_val = d_targets[o * 4 + w];
            uint64_t diff = out_val ^ target_val;
            
            if (w == 3) {
                diff &= ((1ULL << 51) - 1);
            }
            errors_per_bit[o] += __popcll(diff);
        }
    }

    double weighted_fitness = 0.0;
    int raw_total = 0;
    int highest_failing = -1;

    for (int i = 0; i < NUM_OUTPUTS; i++) {
        if (errors_per_bit[i] > 0) {
            if (i > highest_failing) highest_failing = i;
            weighted_fitness += errors_per_bit[i] * pow(10000.0, i);
        }
        raw_total += errors_per_bit[i];
    }

    d_fitnesses[tid].weighted = weighted_fitness;
    d_fitnesses[tid].raw_total = raw_total;
    d_fitnesses[tid].highest_failing = highest_failing;
}

// =====================================================
// Host CPU Logic
// =====================================================
double get_dynamic_mutation_rate(int highest_failing_bit, int stagnation_gens) {
    if (stagnation_gens >= 1000000) return 0.20;
    if (stagnation_gens >= 500000)  return 0.10;
    if (stagnation_gens >= 200000)  return 0.03;
    
    if (highest_failing_bit >= 7) return 0.08;
    if (highest_failing_bit >= 5) return 0.04;
    if (highest_failing_bit >= 3) return 0.01;
    if (highest_failing_bit >= 0) return 0.005;
    return 0.002;
}

void get_active_nodes(const Chromosome& chrom, int current_nodes, bool* active) {
    memset(active, 0, MAX_NODES * sizeof(bool));
    int stack[NUM_OUTPUTS + MAX_NODES * 3];
    int stack_size = 0;

    for (int i = 0; i < NUM_OUTPUTS; i++) {
        stack[stack_size++] = chrom.outputs[i];
    }

    while (stack_size > 0) {
        int gene = stack[--stack_size];
        if (gene < NUM_INPUTS) continue;
        
        int node_idx = gene - NUM_INPUTS;
        if (active[node_idx]) continue;
        
        active[node_idx] = true;
        stack[stack_size++] = chrom.nodes[node_idx].in1;
        stack[stack_size++] = chrom.nodes[node_idx].in2;
        stack[stack_size++] = chrom.nodes[node_idx].in3;
    }
}

void export_verilog(const Chromosome& chrom, int current_nodes, const char* filename) {
    bool active[MAX_NODES];
    get_active_nodes(chrom, current_nodes, active);
    
    ofstream f(filename);
    f << "`timescale 1ns / 1ps\n";
    f << "module unified_input_quantized_activation(\n";
    f << "    input wire [15:0] X_IN,\n";
    f << "    input wire [1:0] MODE,\n";
    f << "    output wire [15:0] Y_OUT\n";
    f << ");\n\n";

    f << "    wire [14:0] abs_x = X_IN[15] ? -X_IN[14:0] : X_IN[14:0];\n\n";
    for (int i = 0; i < 7; i++) f << "    wire W" << i << " = abs_x[" << (i + 6) << "];\n";
    f << "    wire W7 = MODE[0];\n";
    f << "    wire W8 = MODE[1];\n\n";

    for (int i = 0; i < current_nodes; i++) {
        if (active[i]) f << "    wire N" << i << ";\n";
    }
    f << "\n";

    for (int i = 0; i < current_nodes; i++) {
        if (!active[i]) continue;
        Node n = chrom.nodes[i];
        
        char a[32], b[32], c[32];
        sprintf(a, n.in1 < NUM_INPUTS ? "W%d" : "N%d", n.in1 < NUM_INPUTS ? n.in1 : n.in1 - NUM_INPUTS);
        sprintf(b, n.in2 < NUM_INPUTS ? "W%d" : "N%d", n.in2 < NUM_INPUTS ? n.in2 : n.in2 - NUM_INPUTS);
        sprintf(c, n.in3 < NUM_INPUTS ? "W%d" : "N%d", n.in3 < NUM_INPUTS ? n.in3 : n.in3 - NUM_INPUTS);
        
        string op = FUNC_NAMES[n.op];
        
        if (op == "BUF") f << "    assign N" << i << " = " << a << ";\n";
        else if (op == "NOT") f << "    assign N" << i << " = ~" << a << ";\n";
        else if (op == "AND") f << "    assign N" << i << " = " << a << " & " << b << ";\n";
        else if (op == "OR")  f << "    assign N" << i << " = " << a << " | " << b << ";\n";
        else if (op == "XOR") f << "    assign N" << i << " = " << a << " ^ " << b << ";\n";
        else if (op == "NAND")f << "    assign N" << i << " = ~(" << a << " & " << b << ");\n";
        else if (op == "NOR") f << "    assign N" << i << " = ~(" << a << " | " << b << ");\n";
        else if (op == "XNOR")f << "    assign N" << i << " = ~(" << a << " ^ " << b << ");\n";
        else if (op == "MUX") f << "    assign N" << i << " = " << a << " ? " << b << " : " << c << ";\n";
    }

    f << "\n    wire [8:0] cgp_out;\n";
    for (int bit = 0; bit < 9; bit++) {
        int out_node = chrom.outputs[bit];
        if (out_node < NUM_INPUTS) {
            f << "    assign cgp_out[" << bit << "] = W" << out_node << ";\n";
        } else {
            f << "    assign cgp_out[" << bit << "] = N" << (out_node - NUM_INPUTS) << ";\n";
        }
    }

    f << "\n    wire [15:0] raw_residual = {7'd0, cgp_out};\n";
    f << "    wire out_of_bounds = (abs_x > 15'd5120);\n";
    f << "    wire mask_zero_residual = (MODE == 2'b10) || (MODE == 2'b11 && X_IN[15] == 1'b0) || out_of_bounds;\n";
    f << "    wire [15:0] final_residual = mask_zero_residual ? 16'd0 : raw_residual;\n";
    f << "    wire [15:0] relu_base = X_IN[15] ? 16'd0 : X_IN;\n\n";
    f << "    assign Y_OUT = relu_base - final_residual;\n";
    f << "endmodule\n";
    f.close();
    cout << "[+] Verilog mapped and exported to " << filename << endl;
}

// =====================================================
// Main Evolutionary Loop
// =====================================================
int main() {
    // Bind the Ctrl+C signal handler
    signal(SIGINT, int_handler);

    cout << "====================================================\n";
    cout << "Initiating CUDA CGP with Checkpointing (2,187 bits)\n";
    cout << "====================================================\n";

    srand((unsigned)time(NULL));

    // 1. Generate Nano Truth Table using C arrays
    int m_list[243];
    int v_list[243];
    int list_idx = 0;
    int modes[3] = {0, 1, 3};
    
    for (int mi = 0; mi < 3; mi++) {
        for (int step = 0; step < 81; step++) {
            m_list[list_idx] = modes[mi];
            v_list[list_idx] = step << 6;
            list_idx++;
        }
    }

    uint64_t h_inputs[NUM_INPUTS * 4] = {0};
    uint64_t h_targets[NUM_OUTPUTS * 4] = {0};

    for (int i = 0; i < 243; i++) {
        int w = i / 64;
        int b = i % 64;
        int V = v_list[i];
        int M = m_list[i];

        double V_float = V / 1024.0;
        double neg_V = -V_float;
        double y = 0;
        if (M == 0) y = neg_V / (1.0 + exp(-neg_V));
        else if (M == 1) y = 0.5 * neg_V * (1.0 + tanh(sqrt(2.0/M_PI) * (neg_V + 0.044715 * pow(neg_V, 3))));
        else if (M == 3) y = neg_V * tanh(exp(neg_V));

        int target_int = max(0, min(511, (int)round(-y * 1024.0)));

        for (int bit = 0; bit < 9; bit++) {
            if ((target_int >> bit) & 1) h_targets[bit * 4 + w] |= (1ULL << b);
        }

        for (int bit = 0; bit < 7; bit++) {
            if ((V >> (bit + 6)) & 1) h_inputs[bit * 4 + w] |= (1ULL << b);
        }
        if ((M >> 0) & 1) h_inputs[7 * 4 + w] |= (1ULL << b);
        if ((M >> 1) & 1) h_inputs[8 * 4 + w] |= (1ULL << b);
    }

    uint64_t *d_inputs, *d_targets;
    cudaMalloc(&d_inputs, NUM_INPUTS * 4 * sizeof(uint64_t));
    cudaMalloc(&d_targets, NUM_OUTPUTS * 4 * sizeof(uint64_t));
    cudaMemcpy(d_inputs, h_inputs, NUM_INPUTS * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, h_targets, NUM_OUTPUTS * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);

    Chromosome* d_children;
    Fitness* d_fitnesses;
    cudaMalloc(&d_children, LAMBDA * sizeof(Chromosome));
    cudaMalloc(&d_fitnesses, LAMBDA * sizeof(Fitness));

    Chromosome* h_children = new Chromosome[LAMBDA];
    Fitness* h_fitnesses = new Fitness[LAMBDA];

    Chromosome parent;
    Fitness parent_fit;
    int CURRENT_NODES = INITIAL_NODES;
    int start_gen = 0;
    int last_improvement_gen = 0;

    // --- CHECKPOINT LOADER ---
    if (load_checkpoint(CHECKPOINT_FILE, start_gen, parent, parent_fit, last_improvement_gen, CURRENT_NODES)) {
        cout << "\n[+] Resumed from Gen " << start_gen 
             << " | Nodes: " << CURRENT_NODES 
             << " | High Fail Bit: " << parent_fit.highest_failing
             << " | Wrong Bits: " << parent_fit.raw_total << endl;
        start_gen++; 
    } else {
        cout << "\n[+] No valid checkpoint found. Starting completely fresh..." << endl;
        start_gen = 0;              // Force generation reset
        last_improvement_gen = 0;   // Force stagnation reset
        CURRENT_NODES = INITIAL_NODES;
        
        for (int i = 0; i < MAX_NODES; i++) {
            int max_source = NUM_INPUTS + i - 1;
            parent.nodes[i] = { rand() % (max_source + 1), rand() % (max_source + 1), rand() % (max_source + 1), rand() % 9 };
        }
        for (int i = 0; i < NUM_OUTPUTS; i++) {
            parent.outputs[i] = rand() % (NUM_INPUTS + CURRENT_NODES);
        }

        h_children[0] = parent;
        cudaMemcpy(d_children, h_children, sizeof(Chromosome), cudaMemcpyHostToDevice);
        evaluate_population<<<1, 1>>>(d_children, d_inputs, d_targets, d_fitnesses, CURRENT_NODES);
        cudaMemcpy(h_fitnesses, d_fitnesses, sizeof(Fitness), cudaMemcpyDeviceToHost);
        parent_fit = h_fitnesses[0];
    }

    // --- MAIN EVOLUTION LOOP ---
    int gen = start_gen;
    for (; gen < GENERATIONS && keep_running; gen++) {
        int stagnation = gen - last_improvement_gen;
        
        if (stagnation > 100000 && CURRENT_NODES < MAX_NODES) {
            CURRENT_NODES += GROWTH_STEP;
            last_improvement_gen = gen;
            stagnation = 0;
            cout << "\n[+] EXPANDED POOL: " << CURRENT_NODES << " Nodes (Gen " << gen << ")" << endl;
        }

        if (stagnation > 0 && stagnation % 300000 == 0) {
            bool active[MAX_NODES];
            get_active_nodes(parent, CURRENT_NODES, active);
            for (int i = 0; i < CURRENT_NODES; i++) {
                if (!active[i]) {
                    int max_source = NUM_INPUTS + i - 1;
                    parent.nodes[i].in1 = rand() % (max_source + 1);
                    parent.nodes[i].in2 = rand() % (max_source + 1);
                    parent.nodes[i].in3 = rand() % (max_source + 1);
                    parent.nodes[i].op = rand() % 9;
                }
            }
        }

        double mut_rate = get_dynamic_mutation_rate(parent_fit.highest_failing, stagnation);

        for (int c = 0; c < LAMBDA; c++) {
            h_children[c] = parent;
            for (int i = 0; i < CURRENT_NODES; i++) {
                double r = (double)rand() / RAND_MAX;
                if (r < mut_rate) {
                    int max_source = NUM_INPUTS + i - 1;
                    int gene_type = rand() % 4;
                    if (gene_type == 0) h_children[c].nodes[i].in1 = rand() % (max_source+1);
                    else if (gene_type == 1) h_children[c].nodes[i].in2 = rand() % (max_source+1);
                    else if (gene_type == 2) h_children[c].nodes[i].in3 = rand() % (max_source+1);
                    else if (gene_type == 3) h_children[c].nodes[i].op = rand() % 9;
                }
            }
            for (int i = 0; i < NUM_OUTPUTS; i++) {
                double r = (double)rand() / RAND_MAX;
                if (r < mut_rate) {
                    h_children[c].outputs[i] = rand() % (NUM_INPUTS + CURRENT_NODES);
                }
            }
        }

        int blocks = (LAMBDA + 255) / 256;
        cudaMemcpyAsync(d_children, h_children, LAMBDA * sizeof(Chromosome), cudaMemcpyHostToDevice);
        evaluate_population<<<blocks, 256>>>(d_children, d_inputs, d_targets, d_fitnesses, CURRENT_NODES);
        cudaMemcpyAsync(h_fitnesses, d_fitnesses, LAMBDA * sizeof(Fitness), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        for (int c = 0; c < LAMBDA; c++) {
            if (h_fitnesses[c] <= parent_fit) {
                if (h_fitnesses[c].weighted < parent_fit.weighted) last_improvement_gen = gen;
                parent_fit = h_fitnesses[c];
                parent = h_children[c];
            }
        }

        if (gen % 1000 == 0) {
            cout << "Gen " << setw(8) << gen << " | Nodes: " << CURRENT_NODES 
                 << " | High Fail Bit: " << setw(2) << parent_fit.highest_failing 
                 << " | Wrong Bits: " << setw(4) << parent_fit.raw_total 
                 << " / 2187 | Stag: " << stagnation << endl;
        }

        // --- CHECKPOINT SAVER ---
        if (gen > 0 && gen % CHECKPOINT_INTERVAL == 0) {
            save_checkpoint(CHECKPOINT_FILE, gen, parent, parent_fit, last_improvement_gen, CURRENT_NODES);
        }

        if (parent_fit.raw_total == 0) {
            cout << "\n[!!!] TARGET ACCURACY REACHED AT GEN " << gen << " [!!!]\n";
            break;
        }
    }

    if (!keep_running) {
        cout << "\n[!] Manual interruption detected (Ctrl+C). Saving current checkpoint..." << endl;
    }

    // Save final state using the ACTUAL 'gen', not 'GENERATIONS'
    save_checkpoint(CHECKPOINT_FILE, gen, parent, parent_fit, last_improvement_gen, CURRENT_NODES);
    export_verilog(parent, CURRENT_NODES, "unified_cuda_cgp_activation.v");

    delete[] h_children;
    delete[] h_fitnesses;
    cudaFree(d_inputs); 
    cudaFree(d_targets);
    cudaFree(d_children); 
    cudaFree(d_fitnesses);
    return 0;
}
