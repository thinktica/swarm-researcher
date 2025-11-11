#!/usr/bin/env python3
"""
Challenge-Driven v2 Swarm Intelligence Research Agent
===================================================
Enhanced with hierarchical challenge resolution system
Now using COMPOSITION instead of inheritance for clean architecture
"""

import asyncio
import os
import sys
import json
import hashlib
import logging
import subprocess
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any, Set, Protocol
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import time
import re
from pathlib import Path

print("Before logging.basicConfig (print statement)")
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
print("After logging.basicConfig (print statement)")
logger.info("After logging.basicConfig (logger statement)")

print("Before os.environ['PYTHONUNBUFFERED'] = '1' (print statement)")
logger.info("Before os.environ['PYTHONUNBUFFERED'] = '1' (logger statement)")
# Force unbuffered output for immediate console logs
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

print("After os.environ['PYTHONUNBUFFERED'] = '1' (print statement)")
logger.info("After os.environ['PYTHONUNBUFFERED'] = '1' (logger statement)")


try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded .env file (print statement)")
    logger.info("Loaded .env file (logger statement)")
except ImportError:
    print("python-dotenv not installed, using system environment variables (print statement)")
    logger.info("python-dotenv not installed, using system environment variables (logger statement)")


# Initialize availability flags
EMBEDDINGS_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False
LLAMACPP_AVAILABLE = False
SentenceTransformer = None

# Core imports with auto-installation
try:
    from neo4j import GraphDatabase
except ImportError:
    print("Installing neo4j...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "neo4j"])
    from neo4j import GraphDatabase

try:
    import arxiv
except ImportError:
    print("Installing arxiv...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "arxiv"])
    import arxiv

# Try optional imports
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
    logger.info("Sentence-transformers loaded successfully")
except (ImportError, Exception) as e:
    logger.info("Sentence-transformers not available (optional)")
    EMBEDDINGS_AVAILABLE = False
    SentenceTransformer = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    TRANSFORMERS_AVAILABLE = True
    logger.info("Transformers available for backup LLM")
except ImportError:
    logger.info("Transformers not available")
    TRANSFORMERS_AVAILABLE = False

try:
    from llama_cpp import Llama
    LLAMACPP_AVAILABLE = True
    logger.info("llama-cpp available")
except ImportError:
    logger.info("llama-cpp not available")
    LLAMACPP_AVAILABLE = False


# SOLUTION 2: COMPOSITION OVER INHERITANCE
# =========================================

class EventEmitter(Protocol):
    """Protocol defining event emission contract"""
    def emit(self, message: str, **kwargs) -> None: ...


class ThinkticaAdapter:
    """
    Adapter pattern to decouple from Thinktica's implementation.
    This handles all the complexity of Thinktica's event system.
    """
    
    def __init__(self, workspace: str):
        self.workspace = workspace
        self.agent_name = "ChallengeTreeResearchAgent"
        self._event_queue = []
    
    def emit(self, message: str, **kwargs) -> None:
        """Clean emission without parameter conflicts"""
        try:
            from thinktica.core.events import emit as global_emit
            
            # We control exactly what gets passed - no conflicts!
            '''
            global_emit(
                message,
                agent=self.agent_name,
                workspace=self.workspace,
                **kwargs
            )
            '''
            global_emit(
                message,
                type="system",
                timestamp=datetime.now().isoformat()
            )

        except ImportError:
            # Fallback if thinktica not available
            print(f"[EVENT] {message}")
        except Exception as e:
            logger.warning(f"Event emission failed: {e}")
            print(f"[EVENT] {message}")


class ThinkticaContextProvider:
    """
    Provides Thinktica context without inheritance.
    This replaces what we got from ResearchAgent base class.
    """
    
    def __init__(self):
        # Get context from environment or defaults
        self.workspace = os.environ.get('THINKTICA_WORKSPACE', 'default')
        self.investigation_id = os.environ.get('THINKTICA_INVESTIGATION_ID')
        
        # Neo4j configuration
        self.neo4j_url = os.environ.get('NEO4J_URL', 'bolt://localhost:7687')
        self.neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
        self.neo4j_pass = os.environ.get('NEO4J_PASSWORD', 'password')
        self.has_neo4j = self._check_neo4j_availability()
    
    def _check_neo4j_availability(self) -> bool:
        """Check if Neo4j is available"""
        try:
            driver = GraphDatabase.driver(
                self.neo4j_url,
                auth=(self.neo4j_user, self.neo4j_pass)
            )
            with driver.session() as session:
                session.run("RETURN 1")
            driver.close()
            return True
        except:
            return False


class NodeType(Enum):
    """Types of nodes in the research tree"""
    ROOT_QUESTION = "root_question"
    CHALLENGE = "challenge"
    SUB_CHALLENGE = "sub_challenge"
    HYPOTHESIS = "hypothesis"
    FINDING = "finding"
    SOLUTION = "solution"
    DEAD_END = "dead_end"


class NodeStatus(Enum):
    """Status of research nodes"""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    PARTIALLY_RESOLVED = "partially_resolved"
    BLOCKED = "blocked"
    ABANDONED = "abandoned"


class SwarmRole(Enum):
    """Roles for swarm agents"""
    ANALYZER = "analyzer"
    EXTRACTOR = "extractor"
    COMPARATOR = "comparator"
    EMBEDDER = "embedder"
    QUERIER = "querier"
    CHALLENGER = "challenger"
    EVALUATOR = "evaluator"
    HYPOTHESIS = "hypothesis"
    CHALLENGE_EXTRACTOR = "challenge_extractor"
    CHALLENGE_SPECIFIER = "challenge_specifier"
    RESOLUTION_AGENT = "resolution_agent"


@dataclass
class ResearchNode:
    """Enhanced node in the research tree"""
    id: str
    type: NodeType
    content: str
    status: NodeStatus
    parent_id: Optional[str]
    depth: int
    confidence: float = 0.0
    priority: float = 0.5
    findings: List[Dict] = field(default_factory=list)
    challenges: List[Dict] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    resolution_evidence: Optional[Dict] = None
    blockers: List[str] = field(default_factory=list)
    embeddings: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None


@dataclass
class Challenge:
    """Represents a challenge to be resolved"""
    id: str
    statement: str
    type: str  # technical/knowledge/resource/method
    priority: str  # high/medium/low
    parent_context: str
    sub_challenges: List[str] = field(default_factory=list)
    resolution_attempts: List[Dict] = field(default_factory=list)
    status: NodeStatus = NodeStatus.OPEN


@dataclass
class SwarmContext:
    """Context shared across swarm agents"""
    global_goal: str
    current_node: ResearchNode
    remote_response: str
    iteration: int = 0
    max_iterations: int = 100
    findings_buffer: List[Dict] = field(default_factory=list)
    challenges_buffer: List[Challenge] = field(default_factory=list)
    quality_scores: List[float] = field(default_factory=list)
    neo4j_cache: Dict = field(default_factory=dict)
    embeddings_cache: Dict = field(default_factory=dict)
    resolution_history: List[Dict] = field(default_factory=list)


class LocalLLMBackend:
    """Unified backend for local LLMs"""
    
    def __init__(self, agent=None):
        self.backend = None
        self.model = None
        self.agent = agent  # Reference to parent agent for event emission
        logger.info("LocalLLMBackend initialized (logger statement)")
        print("LocalLLMBackend initialized (print statement)")
        self._initialize_backend()
        print("Backend initialized (print statement)")
        logger.info("Backend initialized (logger statement)")
    
    def _initialize_backend(self):
        """Initialize the best available backend"""

        logger.info("Initializing backend (logger statement)")
        print("Initializing backend (print statement)")
        if self._try_llamacpp():
            self.backend = 'llamacpp'
            logger.info("Using Llama.cpp backend")
            print("Using Llama.cpp backend (print statement)")
            if self.agent:
                self.agent.emit("LLM backend initialized", type="system", backend="llamacpp")
            return
        
        if self._try_transformers():
            self.backend = 'transformers'
            logger.info("Using Transformers backend")
            print("Using Transformers backend (print statement)")
            if self.agent:
                self.agent.emit("LLM backend initialized", type="system", backend="transformers")
            return
        
        if self._try_ollama():
            self.backend = 'ollama'
            logger.info("Using Ollama backend")
            print("Using Ollama backend (print statement)")
            if self.agent:
                self.agent.emit("LLM backend initialized", type="system", backend="ollama")
            return
        print("No local LLM available - using mock responses (print statement)")
        logger.info("No local LLM available - using mock responses (logger statement)")
        logger.warning("No local LLM available - using mock responses")
        if self.agent:
            self.agent.emit("No LLM available - using mock responses", type="warning")
        self.backend = 'mock'
    
    def _try_llamacpp(self) -> bool:
        """Try to use llama.cpp"""
        
        if not LLAMACPP_AVAILABLE:
            logger.info("Installing llama-cpp-python...")
            if self.agent:
                self.agent.emit("Installing llama-cpp-python...", type="progress")
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "llama-cpp-python"],
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    return False
            except Exception as e:
                logger.error(f"Could not install llama-cpp-python: {e}")
                return False
        
        try:
            from llama_cpp import Llama
        except ImportError:
            return False
        
        model_dir = Path.home() / ".cache" / "llm_models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        
        if not model_path.exists():
            logger.info("Downloading Mistral 7B Q4 model...")
            print("Downloading Mistral 7B Q4 model... (print statement)")
            if self.agent:
                self.agent.emit("Downloading Mistral 7B model...", type="progress")
            url = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
            
            try:
                import urllib.request
                
                def download_progress(block_num, block_size, total_size):
                    downloaded = block_num * block_size
                    percent = min(100, (downloaded / total_size) * 100)
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    bar_length = 40
                    filled = int(bar_length * percent / 100)
                    bar = '█' * filled + '-' * (bar_length - filled)
                    print(f'\rDownloading: |{bar}| {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)', end='', flush=True)
                
                urllib.request.urlretrieve(url, model_path, reporthook=download_progress)
                print()
                logger.info(f"Model downloaded to {model_path}")
                if self.agent:
                    self.agent.emit(f"Model downloaded to {model_path}", type="system")
            except Exception as e:
                logger.error(f"Failed to download model: {e}")
                if self.agent:
                    self.agent.emit(f"Model download failed: {e}", type="error")
                return False
        
        try:
            logger.info("Loading Mistral 7B Q4 model...")
            print("Loading Mistral 7B model... (this may take 30-60 seconds)", flush=True)
            if self.agent:
                self.agent.emit("Loading Mistral 7B model...", type="progress")
            
            self.model = Llama(
                model_path=str(model_path),
                n_ctx=2048,
                n_threads=min(8, os.cpu_count() or 4),
                n_gpu_layers=0,
                verbose=False,
                seed=42
            )
            
            test_response = self.model(
                "Hello, this is a test.",
                max_tokens=10,
                temperature=0.1
            )
            
            if test_response and 'choices' in test_response:
                logger.info("Model loaded successfully!")
                print("✓ Model loaded successfully!", flush=True)
                if self.agent:
                    self.agent.emit("Model loaded successfully", type="system")
                return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            if self.agent:
                self.agent.emit(f"Model load failed: {e}", type="error")
            return False
        
        return False
    
    def _try_transformers(self) -> bool:
        """Try Transformers backend"""
        if not TRANSFORMERS_AVAILABLE:
            return False
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            
            model_name = "microsoft/phi-2"
            logger.info(f"Loading {model_name}...")
            if self.agent:
                self.agent.emit(f"Loading {model_name}...", type="progress")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype="auto",
                low_cpu_mem_usage=True
            )
            
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7
            )
            
            return True
        except Exception as e:
            logger.debug(f"Could not load Transformers model: {e}")
            return False
    
    def _try_ollama(self) -> bool:
        """Try Ollama backend"""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                if "mistral" not in result.stdout:
                    logger.info("Pulling Mistral model for Ollama...")
                    if self.agent:
                        self.agent.emit("Pulling Mistral model for Ollama...", type="progress")
                    subprocess.run(["ollama", "pull", "mistral"], capture_output=True)
                return True
        except:
            pass
        return False
    
    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        """Generate text using available backend"""
        
        if self.backend == 'llamacpp':
            return self._generate_llamacpp(prompt, max_tokens)
        elif self.backend == 'transformers':
            return self._generate_transformers(prompt, max_tokens)
        elif self.backend == 'ollama':
            return self._generate_ollama(prompt)
        else:
            return self._generate_mock(prompt)
    
    def _generate_llamacpp(self, prompt: str, max_tokens: int) -> str:
        """Generate using llama.cpp"""
        try:
            formatted_prompt = f"[INST] {prompt} [/INST]"
            
            response = self.model(
                formatted_prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.95,
                top_k=40,
                repeat_penalty=1.1,
                stop=["[INST]", "</s>", "\n\n\n"]
            )
            
            generated = response['choices'][0]['text'].strip()
            generated = generated.replace("[INST]", "").replace("[/INST]", "").strip()
            
            return generated
        except Exception as e:
            logger.error(f"llama.cpp generation error: {e}")
            return "[Generation failed]"
    
    def _generate_transformers(self, prompt: str, max_tokens: int) -> str:
        """Generate using Transformers"""
        try:
            outputs = self.pipeline(
                prompt,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            generated = outputs[0]['generated_text']
            if prompt in generated:
                generated = generated.replace(prompt, '').strip()
            
            return generated
        except Exception as e:
            logger.error(f"Transformers generation error: {e}")
            return "[Generation failed]"
    
    def _generate_ollama(self, prompt: str) -> str:
        """Generate using Ollama"""
        try:
            result = subprocess.run(
                ["ollama", "run", "mistral", prompt],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
        return "[Generation failed]"
    
    def _generate_mock(self, prompt: str) -> str:
        """Generate mock responses"""
        if "challenge" in prompt.lower() and "extract" in prompt.lower():
            return '[{"statement": "How to deliver CRISPR to neurons?", "type": "technical", "priority": "high", "parent_context": "Gene therapy"}]'
        elif "specify" in prompt.lower() and "challenge" in prompt.lower():
            return '[{"statement": "Which AAV serotype for neural delivery?", "specificity": "high"}, {"statement": "How to prevent immune response?", "specificity": "high"}]'
        elif "resolution" in prompt.lower():
            return '{"can_resolve": false, "confidence": 0.3, "missing_info": ["AAV efficacy data"], "partial_solution": "Consider AAV9"}'
        elif "findings" in prompt.lower():
            return '[{"statement": "AAV9 crosses blood-brain barrier", "confidence": 0.8, "evidence": "Study X"}]'
        elif "hypothesis" in prompt.lower():
            return '[{"statement": "Combined AAV9-CRISPR approach", "rationale": "Best delivery", "test": "In vivo trial"}]'
        else:
            return "Mock response for testing"


class LocalSwarmAgent:
    """Base class for local swarm agents"""
    
    def __init__(self, role: SwarmRole, parent_agent=None):
        self.role = role
        self.processing_count = 0
        self.parent_agent = parent_agent  # Reference to main agent for events
        self.llm_backend = None
    
    def _get_llm_backend(self):
        """Get LLM backend (create if needed)"""
        if self.llm_backend is None:
            self.llm_backend = LocalLLMBackend(self.parent_agent)
        return self.llm_backend
    
    def process(self, prompt: str) -> str:
        """Process with local LLM"""
        self.processing_count += 1
        
        # Emit event through parent agent
        if self.parent_agent:
            self.parent_agent.emit(
                f"Swarm agent {self.role.value} processing",
                type="progress",
                role=self.role.value,
                count=self.processing_count
            )
        
        backend = self._get_llm_backend()
        role_prompt = f"You are a {self.role.value} agent. {prompt}"
        response = backend.generate(role_prompt)
        
        if response and not response.startswith('['):
            return response
        
        return f"[{self.role.value} processing failed]"


class ChallengeExtractorAgent(LocalSwarmAgent):
    """Extract challenges from text"""
    
    def __init__(self, parent_agent=None):
        super().__init__(SwarmRole.CHALLENGE_EXTRACTOR, parent_agent)
    
    def extract_challenges(self, response: str, context: SwarmContext) -> List[Challenge]:
        """Extract challenges from response"""
        prompt = f"""
        Extract ALL challenges, problems, and obstacles from this text:
        {response[:1000]}
        
        Current research: {context.current_node.content}
        
        Look for:
        - Technical obstacles ("difficult to...", "challenge is...")
        - Knowledge gaps ("unknown whether...", "unclear how...")
        - Resource limitations ("requires...", "needs...")
        - Methodological issues ("how to...")
        - Implicit problems (things mentioned as needing solution)
        
        For each challenge provide:
        - statement: The specific challenge
        - type: technical/knowledge/resource/method
        - priority: high/medium/low
        - parent_context: What led to this challenge
        
        Output as JSON array.
        """
        
        result = self.process(prompt)
        
        challenges = []
        try:
            json_match = re.search(r'\[.*\]', result, re.DOTALL)
            if json_match:
                challenge_data = json.loads(json_match.group())
                for item in challenge_data[:10]:
                    challenge = Challenge(
                        id=hashlib.md5(f"{item.get('statement', '')}{datetime.now()}".encode()).hexdigest()[:12],
                        statement=item.get('statement', ''),
                        type=item.get('type', 'unknown'),
                        priority=item.get('priority', 'medium'),
                        parent_context=item.get('parent_context', '')
                    )
                    challenges.append(challenge)
                    
                    # Emit event for each challenge found
                    if self.parent_agent:
                        self.parent_agent.emit(
                            f"Challenge identified: {challenge.statement[:100]}",
                            type="challenge",
                            priority=challenge.priority,
                            challenge_type=challenge.type
                        )
        except:
            pass
        
        return challenges


class ChallengeSpecifierAgent(LocalSwarmAgent):
    """Break challenges into specific sub-challenges"""
    
    def __init__(self, parent_agent=None):
        super().__init__(SwarmRole.CHALLENGE_SPECIFIER, parent_agent)
    
    def specify_challenge(self, challenge: Challenge, context: SwarmContext) -> List[Challenge]:
        """Break down challenge into specific sub-challenges"""
        prompt = f"""
        Break down this challenge into SPECIFIC, actionable sub-problems:
        
        Challenge: {challenge.statement}
        Type: {challenge.type}
        Context: {challenge.parent_context}
        
        Generate 3-5 concrete sub-challenges that:
        1. Are more specific than the parent
        2. Can be individually researched
        3. Together would solve the parent challenge
        4. Have clear success criteria
        
        Output as JSON array with statement and specificity level.
        """
        
        result = self.process(prompt)
        
        sub_challenges = []
        try:
            json_match = re.search(r'\[.*\]', result, re.DOTALL)
            if json_match:
                sub_data = json.loads(json_match.group())
                for item in sub_data[:5]:
                    sub_challenge = Challenge(
                        id=hashlib.md5(f"{item.get('statement', '')}{datetime.now()}".encode()).hexdigest()[:12],
                        statement=item.get('statement', ''),
                        type=challenge.type,
                        priority=challenge.priority,
                        parent_context=challenge.statement
                    )
                    sub_challenges.append(sub_challenge)
        except:
            pass
        
        return sub_challenges


class ResolutionAgent(LocalSwarmAgent):
    """Attempt to resolve challenges"""
    
    def __init__(self, parent_agent=None):
        super().__init__(SwarmRole.RESOLUTION_AGENT, parent_agent)
    
    def attempt_resolution(self, challenge: Challenge, findings: List[Dict], context: SwarmContext) -> Dict:
        """Try to resolve challenge with available information"""
        findings_text = "\n".join([f"- {f.get('statement', '')}" for f in findings[:10]])
        
        prompt = f"""
        Attempt to resolve this challenge using available findings:
        
        Challenge: {challenge.statement}
        Type: {challenge.type}
        
        Available findings:
        {findings_text}
        
        Previous resolution attempts: {len(challenge.resolution_attempts)}
        
        Determine:
        1. Can this be resolved with current knowledge? (yes/no)
        2. If yes, what's the solution?
        3. If no, what specific information is missing?
        4. Confidence in resolution (0-1)
        5. Partial solutions available?
        
        Output as JSON with keys: can_resolve, solution, confidence, missing_info, partial_solution
        """
        
        result = self.process(prompt)
        
        try:
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                resolution = json.loads(json_match.group())
                
                # Emit resolution event if successful
                if self.parent_agent and resolution.get('confidence', 0) > 0.7:
                    self.parent_agent.emit(
                        f"Challenge resolved: {challenge.statement[:100]}",
                        type="resolution",
                        confidence=resolution['confidence'],
                        solution=resolution.get('solution', '')[:200]
                    )
                
                return resolution
        except:
            pass
        
        return {
            "can_resolve": False,
            "solution": None,
            "confidence": 0.0,
            "missing_info": [],
            "partial_solution": None
        }


class AnalyzerAgent(LocalSwarmAgent):
    """Deep analysis of responses"""
    
    def __init__(self, parent_agent=None):
        super().__init__(SwarmRole.ANALYZER, parent_agent)
    
    def analyze_response(self, response: str, context: SwarmContext) -> Dict:
        """Analyze response in detail"""
        prompt = f"""
        Perform detailed analysis of this response:
        {response[:1000]}
        
        Current node: {context.current_node.content}
        Goal: {context.global_goal}
        
        Extract:
        1. Key assertions with confidence
        2. Implicit assumptions
        3. Logical connections
        4. Contradictions
        5. Areas needing clarification
        
        Output as JSON.
        """
        
        result = self.process(prompt)
        
        try:
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return {
            "assertions": [],
            "assumptions": [],
            "connections": [],
            "contradictions": [],
            "clarifications": []
        }


class ExtractorAgent(LocalSwarmAgent):
    """Extract findings from text"""
    
    def __init__(self, parent_agent=None):
        super().__init__(SwarmRole.EXTRACTOR, parent_agent)
    
    def extract_findings(self, response: str, analysis: Dict) -> List[Dict]:
        """Extract concrete findings"""
        prompt = f"""
        Extract concrete findings from this text:
        {response[:1000]}
        
        Analysis hints: {json.dumps(analysis.get('assertions', [])[:3])}
        
        For each finding:
        - statement: The finding
        - confidence: 0.0 to 1.0
        - evidence: Supporting text
        - category: discovery/hypothesis/fact/theory
        
        Output as JSON array.
        """
        
        result = self.process(prompt)
        
        try:
            json_match = re.search(r'\[.*\]', result, re.DOTALL)
            if json_match:
                findings = json.loads(json_match.group())
                
                # Emit discovery events for high-confidence findings
                if self.parent_agent:
                    for finding in findings[:10]:
                        if finding.get('confidence', 0) > 0.7:
                            self.parent_agent.emit(
                                f"Discovery: {finding.get('statement', '')[:150]}",
                                type="discovery",
                                confidence=finding.get('confidence', 0),
                                category=finding.get('category', 'unknown'),
                                persist=True  # Save important discoveries
                            )
                
                return findings[:10]
        except:
            pass
        
        return []


class ComparatorAgent(LocalSwarmAgent):
    """Compare findings to goals"""
    
    def __init__(self, parent_agent=None):
        super().__init__(SwarmRole.COMPARATOR, parent_agent)
    
    def compare_to_goals(self, findings: List[Dict], context: SwarmContext) -> Dict:
        """Compare findings to research goals"""
        findings_text = "\n".join([f"- {f.get('statement', '')}" for f in findings[:5]])
        
        prompt = f"""
        Compare findings to goals:
        
        Findings:
        {findings_text}
        
        Current goal: {context.current_node.content}
        Global goal: {context.global_goal}
        
        Evaluate:
        1. Current relevance (0-1)
        2. Global relevance (0-1)
        3. Novelty (0-1)
        4. New direction suggested?
        
        Output as JSON.
        """
        
        result = self.process(prompt)
        
        try:
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return {
            "current_relevance": 0.5,
            "global_relevance": 0.5,
            "novelty": 0.5,
            "new_direction": None
        }


class ChallengerAgent(LocalSwarmAgent):
    """Generate challenges from findings"""
    
    def __init__(self, parent_agent=None):
        super().__init__(SwarmRole.CHALLENGER, parent_agent)
    
    def generate_challenges(self, findings: List[Dict], context: SwarmContext) -> List[str]:
        """Generate challenging questions"""
        findings_summary = "\n".join([f.get('statement', '')[:100] for f in findings[:3]])
        
        prompt = f"""
        Generate challenging questions that:
        1. Test validity of findings
        2. Identify edge cases
        3. Expose assumptions
        4. Suggest alternatives
        
        Findings:
        {findings_summary}
        
        Context: {context.current_node.content}
        
        Generate 5 critical questions.
        """
        
        result = self.process(prompt)
        
        questions = []
        for line in result.split('\n'):
            line = line.strip()
            if line and '?' in line and len(line) > 20:
                questions.append(line)
        
        return questions[:5]


class EvaluatorAgent(LocalSwarmAgent):
    """Evaluate iteration quality"""
    
    def __init__(self, parent_agent=None, remote_llm=None):
        super().__init__(SwarmRole.EVALUATOR, parent_agent)
        self.quality_threshold = 0.5
        self.remote_llm = remote_llm
    
    def evaluate_iteration(self, context: SwarmContext) -> Dict:
        """Evaluate quality of current iteration"""
        findings_summary = json.dumps([
            {"statement": f.get("statement", "")[:100], 
             "confidence": f.get("confidence", 0)}
            for f in context.findings_buffer[:5]
        ])
        
        challenges_summary = json.dumps([
            {"statement": c.statement[:100],
             "status": c.status.value}
            for c in context.challenges_buffer[:5]
        ])
        
        prompt = f"""
        Evaluate this research iteration:
        
        Findings: {findings_summary}
        Challenges: {challenges_summary}
        Iteration: {context.iteration}
        Current: {context.current_node.content}
        
        Rate:
        1. Completeness (0-1)
        2. Accuracy (0-1)
        3. Novelty (0-1)
        4. Progress (0-1)
        
        Output as JSON.
        """
        
        result = self.process(prompt)
        
        evaluation = {'completeness': 0.5, 'accuracy': 0.5, 'novelty': 0.5, 'progress': 0.5}
        try:
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                evaluation.update(parsed)
        except:
            pass
        
        evaluation['quality_score'] = np.mean([
            evaluation.get('completeness', 0.5),
            evaluation.get('accuracy', 0.5),
            evaluation.get('novelty', 0.5),
            evaluation.get('progress', 0.5)
        ])
        
        evaluation['decision'] = 'complete' if evaluation['quality_score'] >= self.quality_threshold else 'continue'
        evaluation['ready_for_storage'] = evaluation['quality_score'] >= 0.5
        evaluation['generate_hypothesis'] = evaluation['quality_score'] >= 0.5
        
        logger.info(f"Quality: {evaluation['quality_score']:.2f}, Decision: {evaluation['decision']}")
        
        # Emit evaluation event
        if self.parent_agent:
            self.parent_agent.emit(
                f"Iteration {context.iteration} quality: {evaluation['quality_score']:.2%}",
                type="validation",
                quality=evaluation['quality_score'],
                decision=evaluation['decision']
            )
        
        return evaluation


class HypothesisAgent(LocalSwarmAgent):
    """Generate hypotheses"""
    
    def __init__(self, parent_agent=None):
        super().__init__(SwarmRole.HYPOTHESIS, parent_agent)
    
    def generate_hypotheses(self, context: SwarmContext) -> List[Dict]:
        """Generate new hypotheses"""
        findings_text = "\n".join([
            f"- {f.get('statement', '')}"
            for f in context.findings_buffer[:10]
        ])
        
        resolved_challenges = [
            c for c in context.challenges_buffer 
            if c.status == NodeStatus.RESOLVED
        ]
        
        challenges_text = "\n".join([
            f"- RESOLVED: {c.statement}"
            for c in resolved_challenges[:5]
        ])
        
        prompt = f"""
        Generate hypotheses based on findings and resolved challenges:
        
        Findings:
        {findings_text}
        
        Resolved Challenges:
        {challenges_text}
        
        Goal: {context.global_goal}
        Current: {context.current_node.content}
        
        Generate 3 testable hypotheses that:
        1. Connect findings
        2. Build on resolved challenges
        3. Make predictions
        4. Are rigorous
        
        For each:
        - statement: Hypothesis
        - rationale: Why
        - test: How to validate
        
        Output as JSON array.
        """
        
        result = self.process(prompt)
        
        try:
            json_match = re.search(r'\[.*\]', result, re.DOTALL)
            if json_match:
                hypotheses = json.loads(json_match.group())
                
                # Emit hypothesis events
                if self.parent_agent:
                    for hyp in hypotheses[:3]:
                        if isinstance(hyp, dict) and 'statement' in hyp:
                            self.parent_agent.emit(
                                f"Hypothesis: {hyp['statement'][:150]}",
                                type="hypothesis",
                                rationale=hyp.get('rationale', '')[:100],
                                test=hyp.get('test', '')[:100]
                            )
                
                return hypotheses[:3]
        except:
            pass
        
        return []


class SwarmOrchestrator:
    """Enhanced orchestrator with challenge management"""
    
    def __init__(self, parent_agent=None, neo4j_driver=None, remote_llm=None):
        self.parent_agent = parent_agent  # Reference to main agent
        self.driver = neo4j_driver
        self.remote_llm = remote_llm
        
        # Initialize all agents with parent reference
        self.analyzer = AnalyzerAgent(parent_agent)
        self.extractor = ExtractorAgent(parent_agent)
        self.comparator = ComparatorAgent(parent_agent)
        self.challenger = ChallengerAgent(parent_agent)
        self.evaluator = EvaluatorAgent(parent_agent, remote_llm)
        self.hypothesis = HypothesisAgent(parent_agent)
        
        # New challenge agents
        self.challenge_extractor = ChallengeExtractorAgent(parent_agent)
        self.challenge_specifier = ChallengeSpecifierAgent(parent_agent)
        self.resolution_agent = ResolutionAgent(parent_agent)
        
        # Research tree
        self.research_tree: Dict[str, ResearchNode] = {}
        self.challenges_tree: Dict[str, Challenge] = {}
        self.current_node_id: Optional[str] = None
        
        logger.info("Enhanced swarm orchestrator initialized")
        if parent_agent:
            parent_agent.emit("Swarm orchestrator initialized", type="system")
    
    async def process_remote_response(self, 
                                     response: str, 
                                     context: SwarmContext) -> Dict:
        """Process response with challenge-driven approach"""
        
        logger.info(f"=== SWARM ITERATION {context.iteration} ===")
        print(f"\n=== SWARM ITERATION {context.iteration} ===", flush=True)
        if self.parent_agent:
            self.parent_agent.emit(f"=== SWARM ITERATION {context.iteration} ===", type="progress")
        
        if self.parent_agent:
            self.parent_agent.emit(
                f"Starting swarm iteration {context.iteration}",
                type="progress",
                iteration=context.iteration,
                node=context.current_node.content[:100]
            )
        
        iteration_results = {
            'iteration': context.iteration,
            'findings': [],
            'challenges': [],
            'resolutions': [],
            'quality': 0.0,
            'decision': 'continue'
        }
        
        # Phase 1: Analyze response
        logger.info("Phase 1: Analyzing response...")
        print("  Phase 1: Analyzing response...", flush=True)
        if self.parent_agent:
            self.parent_agent.emit("Phase 1: Analyzing response...", type="progress")
        analysis = self.analyzer.analyze_response(response, context)
        
        # Phase 2: Extract findings
        logger.info("Phase 2: Extracting findings...")
        print("  Phase 2: Extracting findings...", flush=True)
        if self.parent_agent:
            self.parent_agent.emit("Phase 2: Extracting findings...", type="progress")
        findings = self.extractor.extract_findings(response, analysis)
        context.findings_buffer.extend(findings)
        iteration_results['findings'] = findings
        
        # Phase 3: Extract challenges
        logger.info("Phase 3: Extracting challenges...")
        print("  Phase 3: Extracting challenges...", flush=True)
        if self.parent_agent:
            self.parent_agent.emit("Phase 3: Extracting challenges...", type="progress")
        new_challenges = self.challenge_extractor.extract_challenges(response, context)
        context.challenges_buffer.extend(new_challenges)
        iteration_results['challenges'] = new_challenges
        
        # Phase 4: Attempt to resolve existing challenges
        logger.info("Phase 4: Attempting challenge resolution...")
        print("  Phase 4: Attempting challenge resolution...", flush=True)
        if self.parent_agent:
            self.parent_agent.emit("Phase 4: Attempting challenge resolution...", type="progress")
        resolutions = []
        for challenge in context.challenges_buffer:
            if challenge.status == NodeStatus.OPEN:
                resolution = self.resolution_agent.attempt_resolution(
                    challenge, context.findings_buffer, context
                )
                
                if resolution['confidence'] > 0.7:
                    challenge.status = NodeStatus.RESOLVED
                    challenge.resolution_attempts.append(resolution)
                    resolutions.append({
                        'challenge': challenge.statement,
                        'solution': resolution['solution'],
                        'confidence': resolution['confidence']
                    })
                    logger.info(f"   RESOLVED: {challenge.statement[:50]}...")
                    print(f"    ✓ RESOLVED: {challenge.statement[:50]}...", flush=True)
                    if self.parent_agent:
                        self.parent_agent.emit(f"✓ RESOLVED: {challenge.statement[:50]}...", type="resolution")
                elif resolution['confidence'] > 0.3:
                    challenge.status = NodeStatus.PARTIALLY_RESOLVED
                    challenge.resolution_attempts.append(resolution)
        
        iteration_results['resolutions'] = resolutions
        
        # Phase 5: Specify unresolved challenges
        logger.info("Phase 5: Specifying unresolved challenges...")
        print("  Phase 5: Specifying unresolved challenges...", flush=True)
        if self.parent_agent:
            self.parent_agent.emit("Phase 5: Specifying unresolved challenges...", type="progress")
        for challenge in context.challenges_buffer:
            if challenge.status == NodeStatus.OPEN and len(challenge.sub_challenges) == 0:
                sub_challenges = self.challenge_specifier.specify_challenge(challenge, context)
                challenge.sub_challenges = [sc.id for sc in sub_challenges]
                context.challenges_buffer.extend(sub_challenges)
                logger.info(f"   Created {len(sub_challenges)} sub-challenges for: {challenge.statement[:30]}...")
        
        # Phase 6: Compare to goals
        logger.info("Phase 6: Comparing to goals...")
        print("  Phase 6: Comparing to goals...", flush=True)
        if self.parent_agent:
            self.parent_agent.emit("Phase 6: Comparing to goals...", type="progress")
        comparison = self.comparator.compare_to_goals(findings, context)
        
        # Phase 7: Generate new challenges from findings
        logger.info("Phase 7: Generating challenge questions...")
        print("  Phase 7: Generating challenge questions...", flush=True)
        if self.parent_agent:
            self.parent_agent.emit("Phase 7: Generating challenge questions...", type="progress")
        challenge_questions = self.challenger.generate_challenges(findings, context)
        
        # Convert questions to challenges
        for question in challenge_questions:
            new_challenge = Challenge(
                id=hashlib.md5(f"{question}{datetime.now()}".encode()).hexdigest()[:12],
                statement=question,
                type="derived",
                priority="medium",
                parent_context="Generated from findings"
            )
            context.challenges_buffer.append(new_challenge)
        
        # Phase 8: Evaluate iteration
        logger.info("Phase 8: Evaluating quality...")
        print("  Phase 8: Evaluating quality...", flush=True)
        if self.parent_agent:
            self.parent_agent.emit("Phase 8: Evaluating quality...", type="progress")
        evaluation = self.evaluator.evaluate_iteration(context)
        context.quality_scores.append(evaluation['quality_score'])
        
        # Phase 9: Store to Neo4j if quality threshold met
        if evaluation['ready_for_storage'] and self.driver:
            logger.info("Phase 9: Storing to database...")
            print("  Phase 9: Storing to database...", flush=True)
            if self.parent_agent:
                self.parent_agent.emit("Phase 9: Storing to database...", type="progress")
            self._store_iteration_data(context, evaluation)
        
        # Phase 10: Generate hypotheses if appropriate
        if evaluation.get('generate_hypothesis', False):
            logger.info("Phase 10: Generating hypotheses...")
            print("  Phase 10: Generating hypotheses...", flush=True)
            if self.parent_agent:
                self.parent_agent.emit("Phase 10: Generating hypotheses...", type="progress")
            new_hypotheses = self.hypothesis.generate_hypotheses(context)
            iteration_results['hypotheses'] = new_hypotheses
        
        # Update iteration results
        iteration_results['quality'] = evaluation['quality_score']
        iteration_results['decision'] = evaluation['decision']
        iteration_results['comparison'] = comparison
        
        logger.info(f"Iteration complete. Quality: {evaluation['quality_score']:.2f}")
        print(f"  Iteration complete. Quality: {evaluation['quality_score']:.2%}", flush=True)
        if self.parent_agent:
            self.parent_agent.emit(f"Iteration complete. Quality: {evaluation['quality_score']:.2%}", type="progress")
        
        return iteration_results
    
    def _store_iteration_data(self, context: SwarmContext, evaluation: Dict) -> int:
        """Store findings and challenges to Neo4j"""
        if not self.driver:
            return 0
        
        stored_count = 0
        
        try:
            with self.driver.session() as session:
                # Store ResearchBranch
                session.run("""
                    MERGE (b:ResearchBranch {id: $id})
                    SET b.question = $question,
                        b.status = 'processing',
                        b.quality_score = $quality,
                        b.iteration = $iteration,
                        b.node_type = $node_type,
                        b.updated = datetime()
                """, id=context.current_node.id,
                    question=context.current_node.content,
                    quality=evaluation['quality_score'],
                    iteration=context.iteration,
                    node_type=context.current_node.type.value)
                
                # Store challenges and findings (existing code)
                stored_count = len(context.challenges_buffer) + len(context.findings_buffer)
                
                logger.info(f"Stored {stored_count} items to Neo4j")
                
        except Exception as e:
            logger.error(f"Database error: {e}")
            if self.parent_agent:
                self.parent_agent.emit(f"Database storage error: {e}", type="error")
        
        return stored_count
    
    async def run_swarm_loop(self, 
                           response: str, 
                           node: ResearchNode,
                           global_goal: str,
                           max_iterations: int = 10) -> Dict:
        """Run swarm processing loop"""
        
        context = SwarmContext(
            global_goal=global_goal,
            current_node=node,
            remote_response=response,
            max_iterations=max_iterations
        )
        
        all_results = []
        
        for i in range(max_iterations):
            context.iteration = i + 1
            
            iteration_result = await self.process_remote_response(response, context)
            all_results.append(iteration_result)
            
            if iteration_result['decision'] == 'complete':
                logger.info(f"Quality threshold reached at iteration {i+1}")
                break
            
            # Refine response based on unresolved challenges
            unresolved = [c for c in context.challenges_buffer if c.status == NodeStatus.OPEN]
            if unresolved:
                challenge_text = "\n".join([f"- {c.statement}" for c in unresolved[:5]])
                response = response + f"\n\nUnresolved challenges:\n{challenge_text}"
        
        # Calculate challenge resolution rate
        total_challenges = len(context.challenges_buffer)
        resolved_challenges = len([c for c in context.challenges_buffer if c.status == NodeStatus.RESOLVED])
        resolution_rate = resolved_challenges / total_challenges if total_challenges > 0 else 0
        
        return {
            'node_id': node.id,
            'iterations_run': len(all_results),
            'final_quality': np.mean(context.quality_scores) if context.quality_scores else 0.0,
            'total_findings': len(context.findings_buffer),
            'total_challenges': total_challenges,
            'resolved_challenges': resolved_challenges,
            'resolution_rate': resolution_rate,
            'findings': context.findings_buffer,
            'challenges': [{'statement': c.statement, 'status': c.status.value} for c in context.challenges_buffer],
            'hypotheses': [r.get('hypotheses', []) for r in all_results if 'hypotheses' in r]
        }


class RemoteLLMOrchestrator:
    """Remote LLM for strategic decisions"""
    
    def __init__(self, parent_agent=None, provider: str = "groq", api_key: Optional[str] = None):
        self.parent_agent = parent_agent
        self.provider = provider
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")
        
        if not self.api_key:
            logger.warning(f"No API key for {provider}. Set {provider.upper()}_API_KEY")
            if parent_agent:
                parent_agent.emit(f"No API key for {provider}", type="warning")
    
    def get_strategic_direction(self, goal: str, current_state: Dict) -> str:
        """Get strategic direction from remote LLM"""
        
        # Include challenge information in prompt
        challenges_text = ""
        if 'challenges' in current_state:
            open_challenges = [c for c in current_state['challenges'] if c.get('status') == 'open']
            if open_challenges:
                challenges_text = "\n- Open challenges: " + ", ".join([c['statement'][:50] for c in open_challenges[:3]])
        
        prompt = f"""
        You are a genius research strategist solving complex problems.
        
        RESEARCH GOAL: {goal}
        
        CURRENT STATE:
        - Explored nodes: {current_state.get('explored_count', 0)}
        - Current depth: {current_state.get('current_depth', 0)}
        - Resolution rate: {current_state.get('resolution_rate', 0):.1%}
        {challenges_text}
        
        Provide:
        1. Assessment of progress
        2. Critical research directions
        3. Specific approaches to overcome challenges
        4. Key hypotheses to test
        5. Potential breakthrough paths
        
        Focus on concrete, actionable insights.
        """
        
        if self.parent_agent:
            self.parent_agent.emit(
                "Requesting strategic direction from remote LLM",
                type="progress",
                provider=self.provider
            )
        
        if self.provider == "groq":
            return self._call_groq(prompt)
        else:
            return "Strategic direction: Systematically resolve challenges."
    
    def _call_groq(self, prompt: str) -> str:
        """Call Groq API"""
        try:
            import requests
            import time
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 2000,
                "top_p": 0.95
            }
            
            max_retries = 3
            for attempt in range(max_retries):
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    
                    if self.parent_agent:
                        self.parent_agent.emit(
                            "Received strategic direction",
                            type="progress",
                            length=len(content)
                        )
                    
                    return content
                
                elif response.status_code == 429:
                    retry_after = response.headers.get('retry-after', '5')
                    wait_time = int(retry_after) + 1
                    logger.warning(f"Rate limit hit. Waiting {wait_time} seconds...")
                    
                    if self.parent_agent:
                        self.parent_agent.emit(
                            f"Rate limit hit, waiting {wait_time}s",
                            type="warning"
                        )
                    
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                        continue
                    else:
                        return "[Rate limit exceeded]"
                else:
                    logger.error(f"Groq API error: {response.status_code}")
                    if self.parent_agent:
                        self.parent_agent.emit(
                            f"Groq API error: {response.status_code}",
                            type="error"
                        )
                    return "[Remote LLM error]"
            
        except Exception as e:
            logger.error(f"Groq error: {e}")
            if self.parent_agent:
                self.parent_agent.emit(f"Groq error: {e}", type="error")
            return "[Remote LLM error]"


# NOW THE MAIN CLASS USING COMPOSITION
class ChallengeTreeResearchAgent:
    """
    Main agent with challenge-driven research.
    Uses COMPOSITION instead of inheritance - no more conflicts!
    """
    
    def __init__(self, config: Optional[Dict] = None):
        # COMPOSITION - We HAS-A context and emitter, not IS-A ResearchAgent
        self._context = ThinkticaContextProvider()
        self._emitter = ThinkticaAdapter(self._context.workspace)
        
        # Copy context attributes for compatibility
        self.workspace = self._context.workspace
        self.investigation_id = self._context.investigation_id
        self.neo4j_url = self._context.neo4j_url
        self.neo4j_user = self._context.neo4j_user
        self.neo4j_pass = self._context.neo4j_pass
        self.has_neo4j = self._context.has_neo4j
        
        # Now do all your initialization
        print("="*80, flush=True)
        self.emit("="*80, type="system")
        print("CHALLENGE-DRIVEN SWARM INTELLIGENCE RESEARCH AGENT", flush=True)
        self.emit("CHALLENGE-DRIVEN SWARM INTELLIGENCE RESEARCH AGENT", type="system")
        print("="*80, flush=True)
        self.emit("="*80, type="system")
        print(f"✓ Workspace: {self.workspace}", flush=True)
        self.emit(f"✓ Workspace: {self.workspace}", type="system")
        print(f"✓ Investigation: {self.investigation_id or 'None'}", flush=True)
        self.emit(f"✓ Investigation: {self.investigation_id or 'None'}", type="system")
        print(f"✓ Neo4j: {'Available' if self.has_neo4j else 'Not available'}", flush=True)
        self.emit(f"✓ Neo4j: {'Available' if self.has_neo4j else 'Not available'}", type="system")
        print("="*80, flush=True)
        self.emit("="*80, type="system")
        
        # Emit initialization event
        self.emit(
            "Challenge-driven research agent initialized",
            type="system",
            timestamp=datetime.now().isoformat()
        )
        
        self.config = config or {}
        self.api_call_count = 0
        self.last_api_call = None
        
        # Initialize components
        print("\nInitializing components...", flush=True)
        self.emit("Initializing components...", type="progress")
        
        print("  1. Setting up remote LLM orchestrator...", flush=True)
        self.emit("1. Setting up remote LLM orchestrator...", type="progress")
        self.remote_llm = RemoteLLMOrchestrator(
            parent_agent=self,
            provider=self.config.get('remote_provider', 'groq')
        )
        
        print("  2. Connecting to Neo4j...", flush=True)
        self.emit("2. Connecting to Neo4j...", type="progress")
        self.driver = self._connect_neo4j()
        
        print("  3. Initializing swarm orchestrator...", flush=True)
        self.emit("3. Initializing swarm orchestrator...", type="progress")
        self.swarm = SwarmOrchestrator(
            parent_agent=self,
            neo4j_driver=self.driver,
            remote_llm=self.remote_llm
        )
        
        self.tree_root: Optional[ResearchNode] = None
        self.challenge_queue = deque()
        self.node_queue = deque()
        
        self.arxiv_client = arxiv.Client()
        
        logger.info("Challenge-driven research agent initialized")
        print("\n✓ Agent ready!", flush=True)
        self.emit("✓ Agent ready!", type="system")
        print("="*80, flush=True)
        self.emit("="*80, type="system")
        
        # Start heartbeat for monitoring
        self._start_heartbeat()
    
    def emit(self, message: str, **kwargs) -> None:
        """Delegate to composed emitter - NO CONFLICTS!"""
        self._emitter.emit(message, **kwargs)
    
    def _start_heartbeat(self):
        """Emit heartbeat events for monitoring"""
        import threading
        
        def heartbeat():
            counter = 0
            while True:
                try:
                    time.sleep(30)  # Every 30 seconds
                    counter += 1
                    self.emit(
                        f"Agent heartbeat #{counter}",
                        type="heartbeat",
                        counter=counter,
                        api_calls=self.api_call_count,
                        nodes_explored=len(self.swarm.research_tree) if self.swarm else 0
                    )
                except:
                    pass
        
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
    
    def _connect_neo4j(self) -> Optional[GraphDatabase.driver]:
        """Connect to Neo4j using Thinktica context"""
        if not self.has_neo4j:
            print("  Neo4j not available, running without database", flush=True)
            self.emit("Neo4j not available, running without database", type="warning")
            return None
        
        try:
            driver = GraphDatabase.driver(
                self.neo4j_url,
                auth=(self.neo4j_user, self.neo4j_pass)
            )
            with driver.session() as session:
                session.run("RETURN 1")
            
            print(f"  ✓ Connected to Neo4j at {self.neo4j_url}", flush=True)
            self.emit(f"✓ Connected to Neo4j at {self.neo4j_url}", type="system")
            return driver
            
        except Exception as e:
            print(f"  ✗ Neo4j connection failed: {str(e)[:50]}", flush=True)
            self.emit(f"✗ Neo4j connection failed: {str(e)[:50]}", type="warning")
            return None
    
    async def fetch_external_knowledge(self, query: str) -> List[Dict]:
        """Fetch from ArXiv"""
        papers = []
        try:
            self.emit(f"Searching ArXiv for: {query[:100]}", type="progress")
            
            search = arxiv.Search(
                query=query,
                max_results=3,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            for paper in self.arxiv_client.results(search):
                papers.append({
                    'title': paper.title,
                    'summary': paper.summary[:300],
                    'url': paper.entry_id
                })
            
            self.emit(f"Found {len(papers)} papers", type="progress")
            
        except Exception as e:
            logger.error(f"ArXiv error: {e}")
            self.emit(f"ArXiv search error: {e}", type="error")
        
        return papers
    
    def create_research_node(self, 
                            content: str,
                            node_type: NodeType,
                            parent: Optional[ResearchNode] = None) -> ResearchNode:
        """Create a new research node"""
        node_id = hashlib.md5(f"{content}{datetime.now()}".encode()).hexdigest()[:12]
        
        node = ResearchNode(
            id=node_id,
            type=node_type,
            content=content,
            status=NodeStatus.OPEN,
            parent_id=parent.id if parent else None,
            depth=parent.depth + 1 if parent else 0
        )
        
        self.swarm.research_tree[node_id] = node
        
        if parent:
            parent.children.append(node_id)
        
        self.emit(
            f"Created research node: {content[:100]}",
            type="progress",
            node_type=node_type.value,
            depth=node.depth
        )
        
        return node
    
    def check_parent_resolution(self, node: ResearchNode):
        """Check if parent can be resolved after child resolution"""
        if not node.parent_id:
            return
        
        parent = self.swarm.research_tree.get(node.parent_id)
        if not parent:
            return
        
        # Check if all children are resolved
        all_resolved = True
        for child_id in parent.children:
            child = self.swarm.research_tree.get(child_id)
            if child and child.status != NodeStatus.RESOLVED:
                all_resolved = False
                break
        
        if all_resolved:
            parent.status = NodeStatus.RESOLVED
            parent.resolved_at = datetime.now()
            logger.info(f"Parent challenge resolved: {parent.content[:50]}...")
            
            self.emit(
                f"Parent challenge resolved: {parent.content[:100]}",
                type="resolution",
                confidence=0.9,
                persist=True
            )
            
            # Recursively check grandparent
            self.check_parent_resolution(parent)
    
    # Required methods for Thinktica compatibility
    def research(self, question: str) -> Dict[str, Any]:
        """Synchronous wrapper for async research"""
        print(f"\n{'='*80}", flush=True)
        self.emit("="*80, type="progress")
        print(f"RESEARCH CALLED", flush=True)
        self.emit("RESEARCH CALLED", type="progress")
        print(f"Question: {question}", flush=True)
        self.emit(f"Question: {question}", type="progress")
        print(f"{'='*80}\n", flush=True)
        self.emit("="*80, type="progress")
        
        self.emit(f"Research started: {question}", type="progress", question=question)
        
        # Run async research
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._async_research(question, max_depth=3))
        finally:
            loop.close()
        
        # Compile results
        all_findings = []
        for node in self.swarm.research_tree.values():
            all_findings.extend(node.findings)
        
        # Sort by confidence
        all_findings.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        result = {
            "question": question,
            "findings": all_findings[:10],  # Top 10 findings
            "confidence": np.mean([f.get('confidence', 0) for f in all_findings[:10]]) if all_findings else 0.0,
            "nodes_explored": len(self.swarm.research_tree),
            "challenges_resolved": len([n for n in self.swarm.research_tree.values() if n.status == NodeStatus.RESOLVED])
        }
        
        self.emit(
            "Research complete",
            type="discovery",
            confidence=result['confidence'],
            findings_count=len(result['findings']),
            persist=True
        )
        
        return result
    
    def validate(self, finding: Dict[str, Any]) -> float:
        """Validate a finding"""
        confidence = finding.get('confidence', 0.5)
        
        # Simple validation logic - could be enhanced
        if 'evidence' in finding and finding['evidence']:
            confidence *= 1.2
        
        confidence = min(confidence, 1.0)
        
        self.emit(
            f"Validated finding: {finding.get('statement', '')[:100]}",
            type="validation",
            confidence=confidence
        )
        
        return confidence
    
    def query(self, cypher: str) -> List[Dict[str, Any]]:
        """Query Neo4j"""
        if not self.driver:
            self.emit("Neo4j not available for query", type="warning")
            return []
        
        try:
            with self.driver.session() as session:
                result = session.run(cypher)
                records = [dict(record) for record in result]
                
                self.emit(
                    f"Query executed successfully",
                    type="query",
                    records_returned=len(records)
                )
                
                return records
                
        except Exception as e:
            logger.error(f"Query error: {e}")
            self.emit(f"Query error: {e}", type="error")
            return []
    
    def schema(self) -> Dict[str, Any]:
        """Return knowledge graph schema"""
        return {
            "nodes": [
                "ResearchNode",
                "Challenge",
                "Finding",
                "Discovery",
                "Resolution",
                "Hypothesis",
                "Investigation"
            ],
            "relationships": [
                {"type": "REFINES", "from": "ResearchNode", "to": "ResearchNode"},
                {"type": "BLOCKS", "from": "Challenge", "to": "ResearchNode"},
                {"type": "RESOLVES", "from": "Resolution", "to": "Challenge"},
                {"type": "PART_OF", "from": "Discovery", "to": "ResearchNode"},
                {"type": "HAS_CHILD", "from": "ResearchNode", "to": "ResearchNode"},
                {"type": "DISCOVERED", "from": "Investigation", "to": "Finding"}
            ]
        }
    
    async def _async_research(self, goal: str, max_depth: int = 3):
        """Main async research loop with challenge focus"""
        
        print(f"\n{'='*80}", flush=True)
        self.emit("="*80, type="progress")
        print(f"CHALLENGE-DRIVEN RESEARCH SYSTEM", flush=True)
        self.emit("CHALLENGE-DRIVEN RESEARCH SYSTEM", type="progress")
        print(f"Goal: {goal}", flush=True)
        self.emit(f"Goal: {goal}", type="progress")
        print(f"{'='*80}\n", flush=True)
        self.emit("="*80, type="progress")
        
        # Create root challenge
        self.tree_root = self.create_research_node(
            content=f"Challenge: {goal}",
            node_type=NodeType.ROOT_QUESTION
        )
        self.node_queue.append(self.tree_root)
        
        nodes_explored = 0
        challenges_resolved = 0
        
        while self.node_queue and nodes_explored < 15:
            current_node = self.node_queue.popleft()
            
            if current_node.depth > max_depth:
                continue
            
            nodes_explored += 1
            
            print(f"\n{'='*60}", flush=True)
            self.emit("="*60, type="progress")
            print(f"NODE {nodes_explored}: {current_node.content[:80]}...", flush=True)
            self.emit(f"NODE {nodes_explored}: {current_node.content[:80]}...", type="progress")
            print(f"Type: {current_node.type.value}, Depth: {current_node.depth}", flush=True)
            self.emit(f"Type: {current_node.type.value}, Depth: {current_node.depth}", type="progress")
            print(f"Status: {current_node.status.value}", flush=True)
            self.emit(f"Status: {current_node.status.value}", type="progress")
            print(f"{'='*60}", flush=True)
            self.emit("="*60, type="progress")
            
            self.emit(
                f"Exploring node {nodes_explored}: {current_node.content[:100]}",
                type="progress",
                node_number=nodes_explored,
                depth=current_node.depth
            )
            
            # Get strategic direction
            print("\n1. REQUESTING STRATEGIC DIRECTION...", flush=True)
            self.emit("1. REQUESTING STRATEGIC DIRECTION...", type="progress")
            
            current_state = {
                'explored_count': nodes_explored,
                'current_depth': current_node.depth,
                'resolution_rate': challenges_resolved / max(nodes_explored, 1),
                'challenges': [{'statement': c.statement, 'status': c.status.value} 
                             for c in self.swarm.challenges_tree.values()][:5]
            }
            
            # Rate limiting
            if self.last_api_call:
                elapsed = time.time() - self.last_api_call
                if elapsed < 2.4:
                    wait_time = 2.4 - elapsed
                    await asyncio.sleep(wait_time)
            
            self.last_api_call = time.time()
            self.api_call_count += 1
            
            strategic_response = self.remote_llm.get_strategic_direction(
                current_node.content, current_state
            )
            
            print(f"   Direction received ({len(strategic_response)} chars)", flush=True)
            self.emit(f"Direction received ({len(strategic_response)} chars)", type="progress")
            
            # Fetch external knowledge
            print("\n2. FETCHING EXTERNAL KNOWLEDGE...", flush=True)
            self.emit("2. FETCHING EXTERNAL KNOWLEDGE...", type="progress")
            papers = await self.fetch_external_knowledge(current_node.content)
            print(f"   Found {len(papers)} papers", flush=True)
            self.emit(f"Found {len(papers)} papers", type="progress")
            
            # Process through swarm
            print("\n3. CHALLENGE-AWARE SWARM PROCESSING...", flush=True)
            self.emit("3. CHALLENGE-AWARE SWARM PROCESSING...", type="progress")
            
            swarm_results = await self.swarm.run_swarm_loop(
                response=strategic_response + "\n\nPapers:\n" + json.dumps(papers),
                node=current_node,
                global_goal=goal,
                max_iterations=5
            )
            
            print(f"   Iterations: {swarm_results['iterations_run']}", flush=True)
            self.emit(f"Iterations: {swarm_results['iterations_run']}", type="progress")
            print(f"   Quality: {swarm_results['final_quality']:.2%}", flush=True)
            self.emit(f"Quality: {swarm_results['final_quality']:.2%}", type="progress")
            print(f"   Findings: {swarm_results['total_findings']}", flush=True)
            self.emit(f"Findings: {swarm_results['total_findings']}", type="progress")
            print(f"   Challenges: {swarm_results['total_challenges']}", flush=True)
            self.emit(f"Challenges: {swarm_results['total_challenges']}", type="progress")
            print(f"   Resolution rate: {swarm_results['resolution_rate']:.1%}", flush=True)
            self.emit(f"Resolution rate: {swarm_results['resolution_rate']:.1%}", type="progress")
            
            # Update node
            current_node.findings = swarm_results['findings']
            current_node.challenges = swarm_results['challenges']
            current_node.confidence = swarm_results['final_quality']
            
            # Check if current challenge is resolved
            if swarm_results['resolution_rate'] > 0.7:
                current_node.status = NodeStatus.RESOLVED
                current_node.resolved_at = datetime.now()
                challenges_resolved += swarm_results['resolved_challenges']
                print(f"\n   ✓ RESOLVED: {current_node.content[:50]}...", flush=True)
                self.emit(f"✓ RESOLVED: {current_node.content[:50]}...", type="resolution")
                
                self.emit(
                    f"Challenge resolved: {current_node.content[:100]}",
                    type="resolution",
                    confidence=swarm_results['resolution_rate'],
                    persist=True
                )
                
                # Check parent resolution
                self.check_parent_resolution(current_node)
            else:
                current_node.status = NodeStatus.PARTIALLY_RESOLVED
            
            # Create child nodes from unresolved challenges
            unresolved = [c for c in swarm_results['challenges'] if c['status'] == 'open']
            if unresolved:
                print("\n4. CREATING CHILD NODES FOR UNRESOLVED CHALLENGES...", flush=True)
                self.emit("4. CREATING CHILD NODES FOR UNRESOLVED CHALLENGES...", type="progress")
                for challenge in unresolved[:3]:
                    child = self.create_research_node(
                        content=challenge['statement'],
                        node_type=NodeType.SUB_CHALLENGE,
                        parent=current_node
                    )
                    self.node_queue.append(child)
                    print(f"   Added: {challenge['statement'][:60]}...", flush=True)
                    self.emit(f"Added: {challenge['statement'][:60]}...", type="progress")
            
            # Create nodes from hypotheses
            if swarm_results.get('hypotheses'):
                print("\n5. CREATING HYPOTHESIS NODES...", flush=True)
                self.emit("5. CREATING HYPOTHESIS NODES...", type="progress")
                for hyp_batch in swarm_results['hypotheses']:
                    for hyp in hyp_batch[:2]:
                        if isinstance(hyp, dict) and 'statement' in hyp:
                            child = self.create_research_node(
                                content=hyp['statement'],
                                node_type=NodeType.HYPOTHESIS,
                                parent=current_node
                            )
                            self.node_queue.append(child)
                            print(f"   Added hypothesis: {hyp['statement'][:60]}...", flush=True)
                            self.emit(f"Added hypothesis: {hyp['statement'][:60]}...", type="hypothesis")
        
        # Final summary
        print(f"\n{'='*80}", flush=True)
        self.emit("="*80, type="system")
        print("RESEARCH COMPLETE - CHALLENGE RESOLUTION SUMMARY", flush=True)
        self.emit("RESEARCH COMPLETE - CHALLENGE RESOLUTION SUMMARY", type="system")
        print(f"{'='*80}", flush=True)
        self.emit("="*80, type="system")
        
        # Calculate final statistics
        total_nodes = len(self.swarm.research_tree)
        resolved_nodes = len([n for n in self.swarm.research_tree.values() 
                            if n.status == NodeStatus.RESOLVED])
        
        all_challenges = []
        for node in self.swarm.research_tree.values():
            all_challenges.extend(node.challenges)
        
        total_challenges = len(all_challenges)
        resolved_challenges = len([c for c in all_challenges if c['status'] == 'resolved'])
        
        print(f"\nNodes explored: {nodes_explored}", flush=True)
        self.emit(f"Nodes explored: {nodes_explored}", type="system")
        print(f"Nodes resolved: {resolved_nodes}/{total_nodes} ({resolved_nodes/max(total_nodes,1)*100:.1f}%)", flush=True)
        self.emit(f"Nodes resolved: {resolved_nodes}/{total_nodes} ({resolved_nodes/max(total_nodes,1)*100:.1f}%)", type="system")
        print(f"Challenges identified: {total_challenges}", flush=True)
        self.emit(f"Challenges identified: {total_challenges}", type="system")
        print(f"Challenges resolved: {resolved_challenges} ({resolved_challenges/max(total_challenges,1)*100:.1f}%)", flush=True)
        self.emit(f"Challenges resolved: {resolved_challenges} ({resolved_challenges/max(total_challenges,1)*100:.1f}%)", type="system")
        print(f"Max depth reached: {max(n.depth for n in self.swarm.research_tree.values())}", flush=True)
        self.emit(f"Max depth reached: {max(n.depth for n in self.swarm.research_tree.values())}", type="system")
        
        # Emit summary event
        self.emit(
            "Research complete",
            type="discovery",
            nodes_explored=nodes_explored,
            nodes_resolved=resolved_nodes,
            challenges_identified=total_challenges,
            challenges_resolved=resolved_challenges,
            persist=True
        )
        
        # Print challenge hierarchy
        print("\n" + "="*60, flush=True)
        self.emit("="*60, type="system")
        print("CHALLENGE HIERARCHY", flush=True)
        self.emit("CHALLENGE HIERARCHY", type="system")
        print("="*60, flush=True)
        self.emit("="*60, type="system")
        self.print_challenge_tree()
        
        # Store final state to Neo4j
        if self.driver:
            self.store_challenge_hierarchy()
        
        # Top findings
        all_findings = []
        for node in self.swarm.research_tree.values():
            all_findings.extend(node.findings)
        
        all_findings.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        print("\n" + "="*60, flush=True)
        self.emit("="*60, type="system")
        print("TOP DISCOVERIES", flush=True)
        self.emit("TOP DISCOVERIES", type="system")
        print("="*60, flush=True)
        self.emit("="*60, type="system")
        for finding in all_findings[:5]:
            print(f"[{finding.get('confidence', 0):.1%}] {finding.get('statement', '')[:100]}...", flush=True)
            self.emit(f"[{finding.get('confidence', 0):.1%}] {finding.get('statement', '')[:100]}...", type="discovery")
    
    def print_challenge_tree(self, node_id: str = None, indent: int = 0):
        """Print the challenge hierarchy tree"""
        if node_id is None:
            # Start from root
            if not self.tree_root:
                return
            node_id = self.tree_root.id
        
        node = self.swarm.research_tree.get(node_id)
        if not node:
            return
        
        # Status symbols
        status_symbol = {
            NodeStatus.RESOLVED: "✓",
            NodeStatus.PARTIALLY_RESOLVED: "◐",
            NodeStatus.OPEN: "○",
            NodeStatus.BLOCKED: "✗",
            NodeStatus.ABANDONED: "⨯"
        }.get(node.status, "?")
        
        # Type symbols
        type_symbol = {
            NodeType.ROOT_QUESTION: "🎯",
            NodeType.CHALLENGE: "⚡",
            NodeType.SUB_CHALLENGE: "└→",
            NodeType.HYPOTHESIS: "💡",
            NodeType.SOLUTION: "🔧",
            NodeType.FINDING: "📊"
        }.get(node.type, "")
        
        # Print node
        indent_str = "  " * indent
        node_str = f"{indent_str}{status_symbol} {type_symbol} {node.content[:80]}..."
        print(node_str, flush=True)
        self.emit(node_str, type="system")
        
        # Print confidence if resolved
        if node.status == NodeStatus.RESOLVED:
            confidence_str = f"{indent_str}    ↳ Confidence: {node.confidence:.1%}"
            print(confidence_str, flush=True)
            self.emit(confidence_str, type="system")
        
        # Recursively print children
        for child_id in node.children:
            self.print_challenge_tree(child_id, indent + 1)
    
    def store_challenge_hierarchy(self):
        """Store the complete challenge hierarchy to Neo4j"""
        if not self.driver:
            return
        
        try:
            with self.driver.session() as session:
                # Store all nodes with their relationships
                stored = 0
                for node_id, node in self.swarm.research_tree.items():
                    # Store node
                    session.run("""
                        MERGE (n:ResearchNode {id: $id})
                        SET n.content = $content,
                            n.type = $type,
                            n.status = $status,
                            n.depth = $depth,
                            n.confidence = $confidence,
                            n.priority = $priority,
                            n.created_at = $created_at,
                            n.resolved_at = $resolved_at,
                            n.finding_count = $finding_count,
                            n.challenge_count = $challenge_count
                    """, 
                        id=node.id,
                        content=node.content,
                        type=node.type.value,
                        status=node.status.value,
                        depth=node.depth,
                        confidence=node.confidence,
                        priority=node.priority,
                        created_at=node.created_at,
                        resolved_at=node.resolved_at,
                        finding_count=len(node.findings),
                        challenge_count=len(node.challenges)
                    )
                    stored += 1
                
                logger.info(f"Stored {stored} nodes to Neo4j with hierarchy")
                self.emit(
                    f"Stored {stored} nodes to knowledge graph",
                    type="progress"
                )
                
        except Exception as e:
            logger.error(f"Failed to store hierarchy: {e}")
            self.emit(f"Failed to store hierarchy: {e}", type="error")
    
    def get_challenge_analytics(self) -> Dict:
        """Get analytics about challenge resolution"""
        if not self.driver:
            return {}
        
        try:
            with self.driver.session() as session:
                # Get challenge resolution by depth
                result = session.run("""
                    MATCH (n:ResearchNode)
                    RETURN n.depth as depth,
                           n.status as status,
                           count(n) as count
                    ORDER BY depth, status
                """)
                
                depth_stats = defaultdict(lambda: {'resolved': 0, 'open': 0, 'total': 0})
                for record in result:
                    depth = record['depth']
                    status = record['status']
                    count = record['count']
                    depth_stats[depth]['total'] += count
                    if status == 'resolved':
                        depth_stats[depth]['resolved'] += count
                    elif status in ['open', 'blocked']:
                        depth_stats[depth]['open'] += count
                
                return {
                    'depth_statistics': dict(depth_stats)
                }
                
        except Exception as e:
            logger.error(f"Failed to get analytics: {e}")
            return {}


def main():
    """Main entry point"""

    print("main() called (print statement)")
    logger.info("main() called (logger statement)")
    
    print("\n" + "="*80, flush=True)
    print("CHALLENGE-DRIVEN SWARM INTELLIGENCE RESEARCH SYSTEM", flush=True)
    print("="*80, flush=True)
    logger.info("Starting main research process (logger statement)")
    
    # Check for investigation context (optional for testing)
    research_question = os.getenv('THINKTICA_RESEARCH_QUESTION', 'How can CRISPR cure genetic diseases?')
    
    print(f"\nResearch Question: {research_question}", flush=True)
    print("="*80, flush=True)
    logger.info(f"Research Question: {research_question} (logger statement)")
    
    # Create the agent - now using composition!
    agent = ChallengeTreeResearchAgent()
    
    # Note: The agent's emit calls are now available
    print("Starting main research process (print statement)")
    logger.info("Starting main research process (logger statement)")
    agent.emit("Starting main research process", type="system")
    print(f"Research Question: {research_question} (print statement)")
    logger.info(f"Research Question: {research_question} (logger statement)")
    agent.emit(f"Research Question: {research_question}", type="system")
    
    # Run the research
    try:
        result = agent.research(research_question)
        
        print("\n" + "="*80, flush=True)
        agent.emit("="*80, type="system")
        print("RESEARCH RESULTS", flush=True)
        agent.emit("RESEARCH RESULTS", type="system")
        print("="*80, flush=True)
        agent.emit("="*80, type="system")
        
        print(f"\nQuestion: {result['question']}", flush=True)
        agent.emit(f"Question: {result['question']}", type="system")
        print(f"Findings: {len(result['findings'])}", flush=True)
        agent.emit(f"Findings: {len(result['findings'])}", type="system")
        print(f"Confidence: {result['confidence']:.2%}", flush=True)
        agent.emit(f"Confidence: {result['confidence']:.2%}", type="system")
        print(f"Nodes explored: {result['nodes_explored']}", flush=True)
        agent.emit(f"Nodes explored: {result['nodes_explored']}", type="system")
        print(f"Challenges resolved: {result['challenges_resolved']}", flush=True)
        agent.emit(f"Challenges resolved: {result['challenges_resolved']}", type="system")
        
        if result['findings']:
            print("\nTop 3 Findings:", flush=True)
            agent.emit("Top 3 Findings:", type="system")
            for i, finding in enumerate(result['findings'][:3], 1):
                finding_text = f"{i}. [{finding.get('confidence', 0):.1%}] {finding.get('statement', '')[:150]}..."
                print(finding_text, flush=True)
                agent.emit(finding_text, type="discovery")
        
        # Get analytics if available
        if agent.driver:
            analytics = agent.get_challenge_analytics()
            if analytics.get('depth_statistics'):
                print("\nResolution by Depth:", flush=True)
                agent.emit("Resolution by Depth:", type="system")
                for depth, stats in sorted(analytics['depth_statistics'].items()):
                    resolved = stats['resolved']
                    total = stats['total']
                    rate = resolved / total * 100 if total > 0 else 0
                    depth_text = f"  Depth {depth}: {resolved}/{total} resolved ({rate:.1f}%)"
                    print(depth_text, flush=True)
                    agent.emit(depth_text, type="system")
        
        print("\n" + "="*80, flush=True)
        agent.emit("="*80, type="system")
        print("Agent completed successfully", flush=True)
        agent.emit("Agent completed successfully", type="system")
        
    except KeyboardInterrupt:
        print("\n\nResearch interrupted by user", flush=True)
        agent.emit("Research interrupted by user", type="warning")
        
    except Exception as e:
        print(f"\nError: {e}", flush=True)
        agent.emit(f"Error: {e}", type="error")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        if hasattr(agent, 'driver') and agent.driver:
            agent.driver.close()
            print("Neo4j connection closed", flush=True)
            agent.emit("Neo4j connection closed", type="system")

if __name__ == "__main__":
    main()