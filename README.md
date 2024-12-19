# LLM-Finetune Framework

## **Installation**

1. **Clone the repository:**

   ```bash
   git clone https://github.com/nerses28/llm-finetune.git
   cd llm-finetune
   
2. **Set up a virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate

3. **Install:**

   Edit the `install.sh` script and run it.
   ```bash
   ./install.sh
   ```
   Note: To install flash_attn, a prebuilt wheel specialized for a specific setup is used. To ensure proper functionality, the correct prebuilt version must be downloaded.

   ### Example for understanding the flash_attn wheel filename
   Consider the filename:

   ```bash
   flash_attn-2.7.0.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
   ```
   
   This filename can be broken down as follows:

   | **Segment**                | **Description**                                       |
   |----------------------------|-------------------------------------------------------|
   | `2.7.0.post1`              | The version of the `flash_attn` package.              |
   | `+cu12`                    | Cuda version (Built for **CUDA 12**).                 |
   | `torch2.5`                 | Torch version (Compatible with **PyTorch 2.5**).      |
   | `cxx11abiFALSE`            | C++ ABI compatibility (Here flag is `FALSE`).         |
   | `cp312`                    | Python version (Built for **Python 3.12**).           |
   | `linux_x86_64`             | OS version (**Linux** on x86_64 architecture).        |
   ---

## Trainig and Prediction

  Examples of how to run the application can be found in `launch.sh`. 
  The list of parameters and their default values can be found in cli/utils.py and cli/main.py.
  
## Run the API Server
### Server File Structure
The server-side implementation consists of the following files:

- **server_api.json**: The configuration file.
- **server_api.py**: The main script that sets up and runs the FastAPI server. It loads the model, processes requests, and generates responses.
- **server_client_sdk.py**: Provides a client SDK interface to interact with the server.
- **server_request.py**: Contains utility functions to send requests to the server.

### Usage
To launch the API server, use the following command:

```bash
python3 server_api.py /path/to/your/config.json
```
- The server requires a JSON configuration file to specify model paths and optional parameters.  
  Example of a configuration file (`server_api.json`):
  ```json
  {
    "peft_path": "path/to/peft/model", 
    "base_model_path": "path/to/base/model", 
    "port": 8000
  }
  ```
  - `peft_path`: Path to the trained PEFT adapter. Required if specified.
  - `base_model_path`: Path to the base (vanila) model. This is required only if `peft_path` is not provided (If you want to deploy vanila model).
  - `port`: Optional. The port where the server will run (default is `8000`).

An example of how to send a request to the API server can be found in the file server_request.py. To execute the example request, use the following command:
```bash
python3 server_request.py
```
Note: All server-related scripts (server_api.py, server_client_sdk.py, server_request.py) are currently designed for local usage only. If you plan to deploy the server for remote or production use, update sdk/server_api and ensure proper security measures, such as authentication, HTTPS, and firewall settings, are implemented.

## Project Structure (not complete; only some important parts)

```
llm-finetune/

├── cli/                            # Command-line interface scripts
│   ├── main.py                     # Main CLI entry point (defines commands and some of main parameters)
│   ├── utils.py                    # Main CLI Utils file (contains the list of parameters and their default values)
│   ├── scripts/
│   │   ├── train.py                # Training script
│   │   └── predict.py              # Prediction script
│
├── datasets/                       # Data for training and testing
│
├── server_api.json                 # API server configuration
├── server_api.py                   # FastAPI server script
├── server_client_sdk.py            # SDK for interacting with the server API
├── server_request.py               # Example script to make API requests
└── README.md                       # This file
```
