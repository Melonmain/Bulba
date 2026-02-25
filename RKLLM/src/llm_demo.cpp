#include <string.h>
#include <unistd.h>
#include <string>
#include <fstream>
#include <iostream>
#include <csignal>
#include <vector>
#include <sys/socket.h>
#include <netinet/in.h>
#include <thread>
#include <atomic>

#include "rkllm.h"
#include "definitions.hpp"


using namespace std;
LLMHandle llmHandle = nullptr;

typedef struct RequestContext {
    int client_fd;
    std::atomic<bool> processing_complete;
} RequestContext;

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
    RequestContext *ctx = (RequestContext *)userdata;

    if (state == RKLLM_RUN_FINISH)
    {
        printf("\n");
        // Mark processing complete but don't close socket yet
        if (ctx) {
            ctx->processing_complete = true;
        }
    } else if (state == RKLLM_RUN_ERROR) {
        printf("\\run error\n");
        if (ctx) {
            ctx->processing_complete = true;
        }
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

        // send incremental output back to the client if we have a client fd
        if (result->text && ctx) {
            ssize_t sent = send(ctx->client_fd, result->text, strlen(result->text), 0);
            (void)sent;
        }
    }
    return 0;
}

int open_port()
{
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("socket creation failed");
        return 0;
    }

    int opt = 1;
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        perror("setsockopt failed");
        close(server_fd);
        return 0;
    }

    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("bind failed");
        close(server_fd);
        return 0;
    }

    if (listen(server_fd, 1) < 0) {
        perror("listen failed");
        close(server_fd);
        return 0;
    }

    printf("Server listening on port %d\n", PORT);
    return server_fd;
}

void accept_input(int server_fd)
{
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    int client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
    if (client_fd < 0) {
        perror("accept failed");
        close(server_fd);
        return;
    }

    char buffer[1024] = {0};
    ssize_t bytes_read = read(client_fd, buffer, sizeof(buffer) - 1);
    if (bytes_read > 0) {
        buffer[bytes_read] = '\0';  // Null terminate
        printf("Received: %s\n", buffer);

        RKLLMInput rkllm_input;
        memset(&rkllm_input, 0, sizeof(RKLLMInput));  // Initialize all content to 0
        rkllm_input.input_type = RKLLM_INPUT_PROMPT;
        rkllm_input.role = "user";
        rkllm_input.prompt_input = (char *)buffer;
    
        RKLLMInferParam rkllm_infer_params;
        memset(&rkllm_infer_params, 0, sizeof(RKLLMInferParam));  // Initialize all content to 0
        rkllm_infer_params.mode = RKLLM_INFER_GENERATE;
        rkllm_infer_params.keep_history = 0;

        // create a small per-request context so the callback can write back
        RequestContext *ctx = new RequestContext();
        ctx->client_fd = client_fd;
        ctx->processing_complete = false;

        rkllm_run(llmHandle, &rkllm_input, &rkllm_infer_params, ctx);
        
        // Wait for processing to complete
        while (!ctx->processing_complete) {
            usleep(10000);  // Sleep 10ms to avoid busy-waiting
        }
        
        // Now close the socket and cleanup
        close(client_fd);
        delete ctx;
    }
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " model_path\n";
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

    RKLLMInput rkllm_input;
    memset(&rkllm_input, 0, sizeof(RKLLMInput));  // Initialize all content to 0
    
    // Initialize infer parameter struct
    RKLLMInferParam rkllm_infer_params;
    memset(&rkllm_infer_params, 0, sizeof(RKLLMInferParam));  // Initialize all content to 0
    
    // rkllm_load_prompt_cache(llmHandle, "./prompt_cache.bin"); // Load the cached cache

    rkllm_infer_params.mode = RKLLM_INFER_GENERATE;
    // By default, the chat operates in single-turn mode (no context retention)
    // 0 means no history is retained, each query is independent
    rkllm_infer_params.keep_history = 0;

    // The model has a built-in chat template by default, which defines how prompts are formatted  
    // for conversation. Users can modify this template using this function to customize the  
    // system prompt, prefix, and postfix according to their needs.  
    // rkllm_set_chat_template(llmHandle, "", "<｜User｜>", "<｜Assistant｜>");
    
    int port = open_port();
    if (port > 0) {
        while (true) {
            accept_input(port);
        }
    } 
    else {
        std::cerr << "Failed to open port\n";
    }
    return 0;
}