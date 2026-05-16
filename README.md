# 🤖 OS-Level AI Agent (API/offline/personal Runtime connector based)

> **Current implementation is a sample with API key from deployment via lightning AI for inference**

> **A Sophisticated MVP Platform That Transforms Any API Key into an Autonomous OS-Level Intelligent Agent**
---

**DEMO VIDEO (click to play):** In the video the Agent was asked to broswe the github profile YADAV1825 and then write about the person in a blank word document here the mdoel even read the Resume PDF on the website with pypdf2 package of python by downloading and running shell commands on prompting and asking permission of user before running!!

<p align="center">
  <a href="https://www.youtube.com/watch?v=oneWphrOJEc">
    <img 
      src="https://img.youtube.com/vi/oneWphrOJEc/maxresdefault.jpg" 
      alt="Watch Demo"
      width="700"
    />
  </a>
</p>
---

## 📋 Executive Summary

The Opus Desktop AI Agent is a production-ready MVP platform that enables complete desktop automation through keyboard, mouse, and shell command execution. It features self-healing and self-correcting behavior through visual feedback loops, multi-modal task execution, and human-in-the-loop safety mechanisms.

### Key Capabilities

- ✅ Complete desktop automation through keyboard, mouse, and shell command execution
- ✅ Self-healing and self-correcting behavior through visual feedback loops
- ✅ Multi-modal task execution (visual analysis, code execution, file operations)
- ✅ Human-in-the-loop safety mechanisms with user confirmation gates
- ✅ Voice-based task initiation (optional, gracefully falls back to text)
- ✅ Real-time visual verification through automated screenshot analysis
- ✅ Persistent execution logging and history tracking
- ✅ Reflex command optimization for immediate task execution

---

## 🏗️ Core Architecture

### System Components

```
ROOT ENTRY POINT
  └─ main.py
     Initializes AgentApp GUI and starts the event loop

CORE LAYER (core/)
  ├─ agent.py          → DesktopAgent orchestrator
  ├─ llm.py            → LightningClient & OllamaClient (dual API support)
  ├─ prompts.py        → Advanced system prompts with 5 command modes
  ├─ state.py          → AgentState & AgentStep data models
  
TOOLS LAYER (tools/)
  ├─ toolkit.py        → ToolKit dispatcher & router
  ├─ os_tools.py       → System operations (keystrokes, screenshots, apps)
  ├─ file_tools.py     → File operations (read, write, append, list)
  ├─ code_tools.py     → Python code execution with user confirmation
  ├─ browser_tools.py  → Playwright-based browser automation (optional)
  ├─ real_browser_tools.py → Alternative browser implementation
  
UI LAYER (ui/)
  └─ gui.py            → Tkinter-based Picture-in-Picture floating widget

UTILITIES (utils/)
  └─ parsers.py        → JSON extraction with error tolerance

LOGS (logs/)
  └─ Persistent execution history for audit trail
```

### Data Flow Architecture

```
User Input (Voice/Text)
    ↓
AgentApp GUI (Tkinter Widget)
    ↓
DesktopAgent.run_task()
    ↓
LightningClient.chat() / OllamaClient.chat()
    ↓
LLM Response → JSON Parsing
    ↓
ToolKit.execute()
    ↓
    ├─→ OSTools (keyboard, screenshot)
    ├─→ FileTools (file operations)
    ├─→ CodeTools (Python execution)
    └─→ BrowserTools (web automation)
    ↓
Result/Screenshot Feedback
    ↓
History & State Update
    ↓
LOOP or DONE
```

---

## ⚙️ Key Features & Capabilities

### 1. Autonomous Task Execution

The agent can execute complex multi-step tasks independently:

- Open applications by name or keyboard shortcuts
- Navigate web browsers with keyboard commands
- Compose and send emails via Gmail
- Execute shell commands (with user confirmation gate)
- Run Python scripts inline
- Read and write files
- Search and process web content
- Take screenshots for visual feedback
- Press keyboard combinations for system control
- Type text into any focused window

### 2. Self-Healing Through Visual Feedback

The agent implements a sophisticated **"See-Think-Act-Verify"** loop:

**Step 1: Screenshot Analysis**
- Captures current screen state before actions
- Uses vision_analysis field to describe screen context
- Identifies UI elements, buttons, text, and content

**Step 2: Intelligent Action Selection**
- Chooses appropriate keyboard shortcuts or mouse actions
- Uses tab navigation to focus target elements
- Adapts based on visual feedback from previous steps

**Step 3: Verification Step**
- Takes follow-up screenshots to confirm action success
- Compares expected vs. actual state
- Automatically retries with alternative approaches if action fails

**Step 4: Adaptive Recovery**
- If action fails, agent tries different methods
- Can switch between clicking, typing, keyboard shortcuts
- Learns from failure patterns within same execution

### 3. Advanced Command Modes

The agent supports **5 distinct command interpretation modes**:

```
MODE 1: REFLEX COMMANDS (Immediate Direct Actions)
  - Direct user commands bypass planning
  - Examples: "press alt tab", "type hello", "hit enter"
  - Skips plan.txt creation
  - Response time: <1 second
  - Used for: Quick, immediate actions

MODE 2: SCREEN-CONTEXT COMMANDS (Visual Awareness)
  - Commands referencing visible screen content
  - Examples: "play the song you see", "click that button"
  - Mandatory screenshot as first action
  - Full vision_analysis field required
  - Self-healing via visual feedback

MODE 3: MULTI-STEP PLANNING (Complex Tasks)
  - Traditional task execution with numbered steps
  - Creates plan.txt with execution strategy
  - Updates plan with [DONE] markers
  - Used for: Complex multi-step workflows
  - Example: "Open YouTube and search for Faded and play it"

MODE 4: EMAIL COMPOSITION (Gmail Specific)
  - Dedicated mode for email operations
  - Detects account index (/u/0/, /u/1/)
  - Fills To, Subject, Body using keyboard
  - Uses Ctrl+Enter for sending
  - Verifies with screenshot

MODE 5: WINDOW SWITCHING (Alt-Tab Navigation)
  - Advanced Task Switcher menu navigation
  - Multi-step process with visual verification
  - Holds Alt key across multiple steps
  - Captures target window selection
  - Confirms final window focus
```

### 4. Security & Safety Mechanisms

**Built-In Safety Gates:**

1. **User Confirmation for Dangerous Operations**
   - Shell command execution: Shows command preview, requires user approval
   - Python code execution: Shows code snippet, requires confirmation
   - Prevents: Accidental system damage, malware-like behavior

2. **40-Step Safety Threshold**
   - After 40 steps, agent pauses and asks user
   - Dialog: "I have reached 40 steps. Should I keep going?"
   - User can stop at any milestone
   - Prevents: Infinite loops, runaway automation
   - Timeout: 120 seconds for user response

3. **Stop Request Handling**
   - User can stop agent at any time (⏹ button)
   - Agent gracefully handles InterruptedError
   - Browser kept open for manual intervention
   - Process and file states preserved

4. **Command Validation**
   - JSON output strictly validated
   - Malformed responses detected and logged
   - Timeouts on subprocess operations (90 seconds for shell, 60 for Python)
   - Keyboard input sanitization

5. **Vision-Based Verification**
   - Actions verified through screenshot analysis
   - Prevents blind command execution
   - Automatic fallback if action doesn't produce expected result

### 5. API Key Integration

**Dual API Support:**

```
1. LIGHTNING AI (Primary)
   - Cloud-based API endpoint
   - Authentication: Bearer {api_key}/{teamspace}
   - Model: User-specified (default: lightning-ai/gpt-oss-120b)
   - Supports streaming responses
   - Handles both 'choices' and 'candidates' response formats

2. OLLAMA (Local/Alternative)
   - Local model server support
   - Base URL configurable
   - Useful for: Privacy, offline operation, local models
   - Fallback when API unavailable

ENVIRONMENT VARIABLES:
  - LIGHTNING_API_KEY: Your Lightning AI authentication token
  - LIGHTNING_TEAMSPACE: Your project path (e.g., "username/project")
  - MODEL_NAME: Specific model to use
  - Gracefully degrades if env vars missing
```

---

## 🔧 Component Breakdown

### core/agent.py - Orchestration Engine

**DesktopAgent Class** - Primary orchestrator for the autonomous execution loop

**Key Methods:**

- `_is_direct_command()` - Detects reflex and screen-context commands
- `_make_messages()` - Constructs conversation messages for LLM
- `_call_planner()` - Calls LLM to get next action (JSON)
- `_ask_continue()` - Thread-safe dialog box after 40 steps
- `run_task()` - Main execution loop

### core/llm.py - API Clients

**OllamaClient:** Local model server support with streaming
**LightningClient:** Cloud-based API with structured output support

### core/prompts.py - System Instructions

Advanced system prompts containing:
- 13 total available actions
- Output schema specifications
- Reflex command handling
- Screen-context command guidelines
- Human emulation techniques
- Advanced automation techniques

### tools/toolkit.py - Action Dispatcher

Central command router that dispatches actions to appropriate tools (OS, File, Code, Browser)

### tools/os_tools.py - System Operations

- **press_key()** - Single keys, combos, modifier holds, repeated keys
- **type_text()** - Safe character input to focused window
- **take_screenshot()** - PIL ImageGrab for cross-platform compatibility
- **open_app()** - Application launching with special handling
- **open_url()** - URL navigation in default browser
- **search_web()** - Google search queries
- **read_url()** - HTTP GET with timeout and HTML cleaning
- **shell()** - Shell command execution with user confirmation

### tools/file_tools.py - File Operations

- **read_file()** - Read file contents with truncation
- **write_file()** - Create/overwrite files with directory creation
- **append_file()** - Append to files
- **list_dir()** - Directory listing with sorting

### tools/code_tools.py - Python Execution

- **run_python()** - Sandboxed Python execution with user confirmation

### ui/gui.py - User Interface

**AgentApp Class** - Tkinter Picture-in-Picture Widget

**Features:**
- Floating window (always on top)
- Compact size: 400x190 pixels
- Dark theme (Material Design inspired)
- Voice input support (optional)
- Real-time logging
- Focus management
- History browser

---

## 📊 Workflow Examples

### Example 1: Searching GitHub and Writing to Word

**Task:** Search for GitHub user and write comprehensive document

```
[Step 1] Creating plan...
  ACTION: write_file
  REASON: Creating execution plan

[Step 2] Planning...
  ACTION: browser_open
  REASON: Gathering information from GitHub

[Step 3-5] Information gathering...
  ACTION: browser_read
  REASON: Extract profile and repository information

[Step 7-9] Opening Word...
  ACTION: press_key (win) → type_text (Word) → press_key (enter)
  REASON: Launch Word application

[Step 10-14] Document creation...
  ACTION: take_screenshot → press_key (enter) → type_text (content)
  REASON: Create blank document and populate with information

[Step 15] Completion...
  ACTION: done
  REASON: Task complete - all information gathered and documented

TOTAL EXECUTION TIME: ~2 minutes
STEPS COMPLETED: 15
MODE: Multi-step planning with screen verification
```

### Example 2: Reflex Command

```
Task: "press alt tab"

[Step 1] Executing...
  ACTION: press_key
  REASON: Reflex: user said press alt tab
  RESULT: Pressed key combo: alt+tab

[Step 2]
  ACTION: done
  REASON: Reflex command completed

MODE: Reflex (direct command bypass)
EXECUTION TIME: <1 second
```

### Example 3: Screen-Context Command

```
Task: "play the song you see on the screen"

[Step 1]
  ACTION: take_screenshot
  REASON: Screen-context command: need to see what's on screen first

[Step 2]
  ACTION: press_key
  INPUT: space
  VISION_ANALYSIS: YouTube with "Faded - Alan Walker" video page
  REASON: Playing the visible video using spacebar

[Step 3]
  ACTION: take_screenshot
  REASON: Verify the song started playing

[Step 4]
  ACTION: done
  REASON: Verified: Video is now playing

MODE: Screen-context (visual awareness)
EXECUTION TIME: ~5 seconds
```

---

## 🔄 Execution Flow Diagrams

### User Interaction Flow

```
┌─────────────────────────────────────────┐
│  AgentApp GUI (Tkinter)                 │
│  ┌────────────────────────────────────┐ │
│  │ 🎤 [task entry field]  ▶ 📋 ⏹    │ │
│  └────────────────────────────────────┘ │
└────────┬────────────────────────────────┘
         │
         │ Voice/Text Input
         ↓
┌─────────────────────────────────────────┐
│  start_agent()                          │
│  └─ Creates agent_thread                │
│     └─ run_task(state)                  │
└─────────────────────────────────────────┘
```

### Agent Execution Loop

```
┌──────────────────────────────────────────────────┐
│ 1. CHECK STOP REQUEST                            │
│    If stop_requested → break                     │
└────────┬─────────────────────────────────────────┘
         │
┌────────↓─────────────────────────────────────────┐
│ 2. 40-STEP SAFETY CHECK                          │
│    If turn % 40 == 0 → ask user to continue      │
└────────┬─────────────────────────────────────────┘
         │
┌────────↓─────────────────────────────────────────┐
│ 3. DETECT COMMAND MODE                           │
│    ├─ Reflex?        → Skip planning             │
│    ├─ Screen-context?→ Screenshot first          │
│    └─ Complex?       → enforce_plan = True       │
└────────┬─────────────────────────────────────────┘
         │
┌────────↓─────────────────────────────────────────┐
│ 4. CALL PLANNER (LLM)                            │
│    ├─ Prepare messages with history              │
│    ├─ Include screenshots if available           │
│    ├─ Post to Lightning/Ollama API               │
│    └─ Parse JSON response                        │
└────────┬─────────────────────────────────────────┘
         │
┌────────↓─────────────────────────────────────────┐
│ 5. EXTRACT ACTION                                │
│    ├─ If action == "done" → break                │
│    ├─ Else → prepare for execution               │
│    └─ Log action, reason, vision_analysis        │
└────────┬─────────────────────────────────────────┘
         │
┌────────↓─────────────────────────────────────────┐
│ 6. EXECUTE ACTION                                │
│    ├─ toolkit.execute(action)                    │
│    ├─ Route to OS/File/Code/Browser tools        │
│    ├─ Apply user confirmations if needed         │
│    └─ Capture result/screenshot                  │
└────────┬─────────────────────────────────────────┘
         │
┌────────↓─────────────────────────────────────────┐
│ 7. UPDATE STATE                                  │
│    ├─ Add step to history                        │
│    ├─ Update task with result context            │
│    ├─ Log result                                 │
│    └─ Increment turn counter                     │
└────────┬─────────────────────────────────────────┘
         │
         └─────────────────→ LOOP TO STEP 1
```

### Toolkit Dispatch Flow

```
┌──────────────────────────────────────┐
│ toolkit.execute(action_dict)         │
└────────┬─────────────────────────────┘
         │
         ├─→ action == "open_app"
         │   └─→ OSTools.open_app(name)
         │
         ├─→ action == "press_key"
         │   └─→ OSTools.press_key(key)
         │
         ├─→ action == "take_screenshot"
         │   └─→ OSTools.take_screenshot()
         │       Returns: [SCREENSHOT_TAKEN: path]
         │
         ├─→ action == "shell"
         │   ├─→ Show confirmation dialog
         │   └─→ OSTools.shell(cmd)
         │
         ├─→ action == "run_python"
         │   ├─→ Show confirmation dialog
         │   └─→ CodeTools.run_python(code)
         │
         ├─→ action == "read_file"
         │   └─→ FileTools.read_file(path)
         │
         ├─→ action == "write_file"
         │   └─→ FileTools.write_file(path, content)
         │
         └─→ action == "done"
             └─→ Return "done"
```

---

## 📈 Performance Characteristics

### Latency Profile

```
Task Initiation:
  Voice Recognition: 2-5 seconds (depends on speech length)
  Text Entry: <1 second
  Agent Thread Start: <100ms

Planning Phase (First Step):
  LLM Inference: 3-8 seconds (depends on model, system load)
  JSON Parsing: <100ms

Action Execution:
  Keyboard Operations: 50-200ms
  Screenshot: 1500-2000ms (includes 1.5s delay for stability)
  File Operations: <100ms
  Shell Commands: 1-60 seconds (variable)

Verification Phase:
  Screenshot Analysis: 2-5 seconds (LLM time)
  Decision Making: 3-5 seconds (LLM time)

Total Loop Time (average):
  Simple Action: 8-15 seconds
  Complex Action: 15-30 seconds
  Multi-step Task: 5-10 minutes (depending on task complexity)

Throughput:
  Maximum steps per minute: 4-8 steps/min
  Effective task completion rate: 1-2 complex tasks per hour
```

### Resource Usage

```
Memory:
  Baseline: ~80MB (GUI + core)
  Per 100 steps history: ~10MB
  Screenshot buffer: ~5MB
  Total typical: 100-150MB

CPU:
  Idle: <1% (event-driven)
  During LLM inference: 20-40% (single core)
  During image capture: 10-20%

Network:
  Per API call: ~50KB (varies with screenshot size)
  Screenshots: 200KB-2MB per image
  Typical session: 2-10MB data transfer

Disk:
  Logs per run: 50KB-500KB
  Screenshots: 1-3MB per execution
  Retention: Depends on log cleanup frequency
```

---

## 🔐 Security & Safety Analysis

### Threat Model Mitigation

```
THREAT 1: Malicious shell commands
  ├─ MITIGATION: User confirmation gate on all shell() calls
  ├─ PREVIEW: Command preview (150 chars) shown before execution
  ├─ TIMEOUT: 90-second hard limit prevents long-running exploits
  └─ AUDIT: Full command logged in execution history

THREAT 2: Runaway infinite loops
  ├─ MITIGATION: 40-step safety threshold
  ├─ USER CHECK: Dialog every 40 steps asking to continue
  ├─ TIMEOUT: 120-second timeout for user response
  ├─ EARLY STOP: "done" action checked immediately
  └─ LOG: Step count tracked and reported

THREAT 3: Accidental file deletion
  ├─ MITIGATION: No automatic deletion (write, append, read only)
  ├─ SCOPE: File operations limited to user directories
  ├─ VISIBILITY: All file operations logged
  └─ VERIFICATION: No wildcard deletion patterns

THREAT 4: Unauthorized code execution
  ├─ MITIGATION: run_python requires user confirmation
  ├─ PREVIEW: Code snippet preview (200 chars) shown
  ├─ TIMEOUT: 60-second limit on Python execution
  ├─ SANDBOX: Subprocess isolation
  └─ OUTPUT: Stderr/stdout captured separately

THREAT 5: Browser-based CSRF/XSS
  ├─ MITIGATION: Browser tools optional (graceful fallback)
  ├─ SCOPE: Limited to browser automation, no DOM access
  ├─ ISOLATION: Separate browser instance
  └─ LOG: All browser actions logged

THREAT 6: Stop request not working
  ├─ MITIGATION: Stop check on every iteration
  ├─ SUBPROCESS: Process.kill() on stop
  ├─ LLM: Response stream interrupted
  └─ FEEDBACK: InterruptedError caught and logged
```

### Privacy Considerations

- **Data Collection:** Screenshots sent to LLM for vision analysis
- **History Logging:** All steps logged to disk in ./logs/ directory
- **Network Transmission:** HTTPS encryption used for API calls
- **Recommendations:** Review sensitive data before screenshots, use local Ollama for sensitive tasks

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8+
- API key (Lightning AI or Ollama)

### Installation Steps

1. **Clone or download the project**

2. **Create .env file** in project root:
   ```
   LIGHTNING_API_KEY=your_api_key_here
   LIGHTNING_TEAMSPACE=username/project
   MODEL_NAME=lightning-ai/gpt-oss-120b
   ```

3. **Install dependencies:**
   ```bash
   pip install requests pillow pyautogui python-dotenv
   pip install playwright speech_recognition  # optional
   pip install beautifulsoup4  # optional but recommended
   ```

4. **Run application:**
   ```bash
   python main.py
   ```

---

## 📋 MVP Completion Checklist

```
✅ Core Functionality
  ✓ Autonomous task execution
  ✓ Multi-step planning
  ✓ Screenshot-based visual feedback
  ✓ Self-healing through verification
  ✓ Error handling and recovery

✅ Security Features
  ✓ User confirmation gates (shell, Python)
  ✓ 40-step safety threshold
  ✓ Stop request handling
  ✓ Timeout management
  ✓ Vision-based verification

✅ API Integration
  ✓ Lightning AI client
  ✓ Ollama client (fallback)
  ✓ Environment variable configuration
  ✓ Streaming response handling

✅ Desktop Automation
  ✓ Keyboard control
  ✓ Screenshot capture
  ✓ Application launching
  ✓ URL navigation
  ✓ File operations
  ✓ Shell commands
  ✓ Python execution

✅ User Interface
  ✓ Tkinter GUI widget
  ✓ Real-time logging
  ✓ Voice input (optional)
  ✓ Start/Stop controls
  ✓ Persistent history

✅ Advanced Features
  ✓ Reflex command mode
  ✓ Screen-context mode
  ✓ Multi-step planning mode
  ✓ Email composition mode
  ✓ Window switching mode
  ✓ Base64 screenshot encoding
  ✓ Focus management

✅ Production Ready
  ✓ Error logging
  ✓ Execution history
  ✓ Timeout handling
  ✓ Cross-platform support
  ✓ Graceful degradation for optional features
  ✓ Thread-safe GUI operations
```

---

## 🎯 Deployment Options

### Development
- Run main.py directly
- GUI window visible
- Full logging enabled
- Interactive debugging

### Production (Server)
- Run with headless mode
- GUI root = None
- Batch task mode
- Auto-continues past 40-step check

### Batch Automation
- Create task queue
- Pass multiple tasks
- Logs to persistent storage
- Email/webhook on completion

### Docker Container
- Include all dependencies
- Volume mount for logs
- API key via environment
- xvfb or virtual display for screenshots

---

## 💡 Future Enhancement Opportunities

### Short Term (v1.1)
- Multi-Agent Coordination
- Advanced Vision (OCR, object detection)
- Performance Optimization
- Enhanced Logging

### Medium Term (v2.0)
- Model Switching
- Tool Extension (database, API calls)
- Learning & Adaptation
- Multi-Modal Input

### Long Term (v3.0)
- Mobile Support
- Natural Language Enhancement
- Advanced Safety (Explainability, rollback)
- Enterprise Features (RBAC, audit compliance)

---

## 📊 Technical Specifications

```
Project: Opus Desktop AI Agent
Status: MVP - Production Ready
Language: Python 3.8+
Main Dependencies: requests, pillow, pyautogui, python-dotenv
Optional: playwright, speech_recognition, beautifulsoup4

Core Architecture: Modular, Event-Driven, Thread-Based
UI Framework: Tkinter (cross-platform)
API Support: Lightning AI + Ollama (fallback)
Platform: Windows (primary), macOS/Linux (supported)

Total Lines of Code: ~2500
Modules: 8 core + tools
Config Files: .env
Logging: Persistent (./logs/)
History: Full execution traces with timestamps
```

---

## 📞 Support & Contribution

This project demonstrates a complete autonomous desktop agent implementation suitable for:
- Automated testing scenarios
- Data entry and processing
- Research and information gathering
- Browser automation tasks
- System administration tasks
- Integration testing
- Workflow automation

---

## 📝 Built By

**Rohit Yadav**  
*NIT Jalandhar*

---

## 📄 License

(As specified by author)

---

## 🙏 Acknowledgments

This MVP represents a complete, modular, and extensible architecture ready for production deployment or further enhancement based on specific use case requirements.

**Key Achievements:**
- 13 distinct action types
- 5 command interpretation modes
- 40-step safety threshold
- Vision-based self-healing
- Multi-threaded GUI with real-time logging
- Cross-platform desktop control
- Dual LLM API support
- Comprehensive error recovery

---

**Status:** ✅ Production-Ready MVP  
**Last Updated:** 2026

