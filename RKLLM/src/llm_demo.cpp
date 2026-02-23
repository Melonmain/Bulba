#include <string.h>
#include <unistd.h>
#include <string>
#include <fstream>
#include <iostream>
#include <csignal>
#include <vector>

#include "rkllm.h"
#include "definitions.hpp"


using namespace std;
LLMHandle llmHandle = nullptr;

void exit_handler(int signal)
{
    if (llmHandle != nullptr)
    {
        {
            LLMHandle _tmp = llmHandle;
            llmHandle = nullptr;
            rkllm_destroy(_tmp);
        }
    }
    exit(signal);
}

int callback(RKLLMResult *result, void *userdata, LLMCallState state)
{
    if (state == RKLLM_RUN_FINISH)
    {
        printf("\n");
    } else if (state == RKLLM_RUN_ERROR) {
        printf("\\run error\n");
    } else if (state == RKLLM_RUN_NORMAL) {
        /* ================================================================================================================
        If using GET_LAST_HIDDEN_LAYER functionality, the callback interface will return memory pointer: last_hidden_layer,
        token count: num_tokens, and hidden layer size: embd_size. Through these three parameters, you can obtain the data
        in last_hidden_layer. Note: you need to get it in the current callback. If not obtained in time, the next callback
        will release this pointer.
        ===============================================================================================================*/
        if (result->last_hidden_layer.embd_size != 0 && result->last_hidden_layer.num_tokens != 0) {
            int data_size = result->last_hidden_layer.embd_size * result->last_hidden_layer.num_tokens * sizeof(float);
            printf("\ndata_size:%d",data_size);
            std::ofstream outFile("last_hidden_layer.bin", std::ios::binary);
            if (outFile.is_open()) {
                outFile.write(reinterpret_cast<const char*>(result->last_hidden_layer.hidden_states), data_size);
                outFile.close();
                std::cout << "Data saved to output.bin successfully!" << std::endl;
            } else {
                std::cerr << "Failed to open the file for writing!" << std::endl;
            }
        }
        printf("%s", result->text);
    }
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " model_path max_new_tokens max_context_len\n";
        return 1;
    }

    signal(SIGINT, exit_handler);
    printf("rkllm init start\n");

    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path = argv[1];

    //added by me
    param.extend_param.base_domain_id = 0;
    param.extend_param.embed_flash = 1;

    // Set sampling parameters
    param.top_k = 1;
    param.top_p = 0.95;
    param.temperature = 0.8;
    param.repeat_penalty = 1.1;
    param.frequency_penalty = 0.0;
    param.presence_penalty = 0.0;

    param.max_new_tokens = MAX_TOKENS;
    param.max_context_len = MAX_CONTEXT_SIZE;
    param.skip_special_token = true;
    param.extend_param.base_domain_id = 0;
    param.extend_param.embed_flash = 1;

    int ret = rkllm_init(&llmHandle, &param, callback);
    if (ret == 0){
        printf("rkllm init success\n");
    } else {
        printf("rkllm init failed\n");
        exit_handler(-1);
    }

    vector<string> pre_input;
    pre_input.push_back("There is a cage with some chickens and rabbits. Counting, there are 14 heads and 38 legs in total. How many chickens and rabbits are there?");
    pre_input.push_back("28 children are arranged in a line. The 10th from the left is Xuedou. What position is he from the right?");
    cout << "\n**********************You can enter the corresponding number of questions below to get answers / or custom input********************\n"
         << endl;
    for (int i = 0; i < (int)pre_input.size(); i++)
    {
        cout << "[" << i << "] " << pre_input[i] << endl;
    }
    cout << "\n*************************************************************************\n"
         << endl;

    RKLLMInput rkllm_input;
    memset(&rkllm_input, 0, sizeof(RKLLMInput));  // Initialize all content to 0
    
    // Initialize infer parameter struct
    RKLLMInferParam rkllm_infer_params;
    memset(&rkllm_infer_params, 0, sizeof(RKLLMInferParam));  // Initialize all content to 0

    // 1. Initialize and set LoRA parameters (if you need to use LoRA)
    // RKLLMLoraAdapter lora_adapter;
    // memset(&lora_adapter, 0, sizeof(RKLLMLoraAdapter));
    // lora_adapter.lora_adapter_path = "qwen0.5b_fp16_lora.rkllm";
    // lora_adapter.lora_adapter_name = "test";
    // lora_adapter.scale = 1.0;
    // ret = rkllm_load_lora(llmHandle, &lora_adapter);
    // if (ret != 0) {
    //     printf("\nload lora failed\n");
    // }

    // Load second LoRA
    // lora_adapter.lora_adapter_path = "Qwen2-0.5B-Instruct-all-rank8-F16-LoRA.gguf";
    // lora_adapter.lora_adapter_name = "knowledge_old";
    // lora_adapter.scale = 1.0;
    // ret = rkllm_load_lora(llmHandle, &lora_adapter);
    // if (ret != 0) {
    //     printf("\nload lora failed\n");
    // }

    // RKLLMLoraParam lora_params;
    // lora_params.lora_adapter_name = "test";  // Specify the LoRA name for inference
    // rkllm_infer_params.lora_params = &lora_params;

    // 2. Initialize and set Prompt Cache parameters (if you need to use prompt cache)
    // RKLLMPromptCacheParam prompt_cache_params;
    // prompt_cache_params.save_prompt_cache = true;                  // Whether to save prompt cache
    // prompt_cache_params.prompt_cache_path = "./prompt_cache.bin";  // If you need to save prompt cache, specify the cache file path
    // rkllm_infer_params.prompt_cache_params = &prompt_cache_params;
    
    // rkllm_load_prompt_cache(llmHandle, "./prompt_cache.bin"); // Load the cached cache

    rkllm_infer_params.mode = RKLLM_INFER_GENERATE;
    // By default, the chat operates in single-turn mode (no context retention)
    // 0 means no history is retained, each query is independent
    rkllm_infer_params.keep_history = 0;

    // The model has a built-in chat template by default, which defines how prompts are formatted  
    // for conversation. Users can modify this template using this function to customize the  
    // system prompt, prefix, and postfix according to their needs.  
    // rkllm_set_chat_template(llmHandle, "", "<｜User｜>", "<｜Assistant｜>");
    
    while (true)
    {
        std::string input_str;
        printf("\n");
        printf("user: ");
        std::getline(std::cin, input_str);
        if (input_str == "exit")
        {
            break;
        }
        if (input_str == "clear")
        {
            ret = rkllm_clear_kv_cache(llmHandle, 1, nullptr, nullptr);
            if (ret != 0)
            {
                printf("clear kv cache failed!\n");
            }
            continue;
        }
        for (int i = 0; i < (int)pre_input.size(); i++)
        {
            if (input_str == to_string(i))
            {
                input_str = pre_input[i];
                cout << input_str << endl;
            }
        }
        rkllm_input.input_type = RKLLM_INPUT_PROMPT;
        rkllm_input.role = "user";
        rkllm_input.prompt_input = (char *)input_str.c_str();
        printf("robot: ");

        // To use normal inference functionality, configure rkllm_infer_mode to RKLLM_INFER_GENERATE or do not configure parameters
        rkllm_run(llmHandle, &rkllm_input, &rkllm_infer_params, NULL);
    }
    rkllm_destroy(llmHandle);

    return 0;
}