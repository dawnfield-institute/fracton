"""
ShadowPuppet Mock Generator

Template-based code generator that produces WORKING implementations.
No AI required - generates real, functional code based on protocol specs.
"""

from typing import Dict, Optional, List
import random

from .base import CodeGenerator, GenerationContext


class MockGenerator(CodeGenerator):
    """
    Template-based code generator that produces working implementations.
    
    Generates real, functional code based on protocol specifications.
    No AI required - useful for testing, demos, and as fallback.
    
    Example:
        generator = MockGenerator()
        code = generator.generate(GenerationContext(protocol=api_protocol))
    """
    
    def __init__(self, templates: Optional[Dict[str, str]] = None):
        """
        Initialize mock generator.
        
        Args:
            templates: Custom templates by protocol name
        """
        self.templates = templates or {}
    
    @property
    def name(self) -> str:
        return "mock"
    
    def generate(self, context: GenerationContext) -> str:
        """Generate working code from protocol."""
        protocol = context.protocol
        
        # Check for custom template first
        if protocol.name in self.templates:
            return self.templates[protocol.name]
        
        # Check for known protocol patterns
        name_lower = protocol.name.lower()
        
        if 'router' in name_lower or ('api' in name_lower and 'router' not in name_lower):
            return self._generate_api_router(context)
        elif 'user' in name_lower and 'service' in name_lower:
            return self._generate_user_service(context)
        elif 'template' in name_lower or 'render' in name_lower:
            return self._generate_template_renderer(context)
        elif 'static' in name_lower or ('file' in name_lower and 'server' in name_lower):
            return self._generate_static_server(context)
        elif 'webapp' in name_lower or (name_lower == 'app'):
            return self._generate_webapp(context)
        elif 'intent' in name_lower or 'classif' in name_lower:
            return self._generate_intent_classifier(context)
        elif 'response' in name_lower and 'generat' in name_lower:
            return self._generate_response_generator(context)
        elif 'conversation' in name_lower or ('manager' in name_lower and 'chat' not in name_lower):
            return self._generate_conversation_manager(context)
        elif 'chat' in name_lower and 'bot' in name_lower:
            return self._generate_chatbot(context)
        else:
            return self._generate_generic(context)
    
    def _generate_api_router(self, context: GenerationContext) -> str:
        """Generate a working API router."""
        return '''"""
APIRouter - REST API router with CRUD operations.
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
import re


@dataclass
class Request:
    """HTTP request representation."""
    method: str
    path: str
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[Dict[str, Any]] = None
    query_params: Optional[Dict[str, str]] = None
    path_params: Dict[str, str] = field(default_factory=dict)


@dataclass
class Response:
    """HTTP response representation."""
    status_code: int
    headers: Dict[str, str] = field(default_factory=dict)
    body: Any = None
    
    @staticmethod
    def json(data: Any, status: int = 200) -> 'Response':
        return Response(status, {'Content-Type': 'application/json'}, data)
    
    @staticmethod
    def error(message: str, status: int = 400) -> 'Response':
        return Response.json({'error': message}, status)


class APIRouter:
    """REST API router with CRUD operations."""
    
    def __init__(self):
        self.routes: Dict[str, Dict[str, Callable]] = {'GET': {}, 'POST': {}, 'PUT': {}, 'DELETE': {}}
        self.middleware: List[Callable] = []
    
    def get(self, path: str) -> Callable:
        def decorator(handler: Callable) -> Callable:
            self.routes['GET'][path] = handler
            return handler
        return decorator
    
    def post(self, path: str) -> Callable:
        def decorator(handler: Callable) -> Callable:
            self.routes['POST'][path] = handler
            return handler
        return decorator
    
    def put(self, path: str) -> Callable:
        def decorator(handler: Callable) -> Callable:
            self.routes['PUT'][path] = handler
            return handler
        return decorator
    
    def delete(self, path: str) -> Callable:
        def decorator(handler: Callable) -> Callable:
            self.routes['DELETE'][path] = handler
            return handler
        return decorator
    
    def use(self, middleware: Callable) -> None:
        self.middleware.append(middleware)
    
    def _match_route(self, method: str, path: str) -> tuple:
        routes = self.routes.get(method, {})
        if path in routes:
            return routes[path], {}
        for route_path, handler in routes.items():
            pattern = re.sub(r':(\w+)', r'(?P<\\1>[^/]+)', route_path)
            match = re.fullmatch(pattern, path)
            if match:
                return handler, match.groupdict()
        return None, {}
    
    def handle(self, request: Request) -> Response:
        for mw in self.middleware:
            result = mw(request)
            if isinstance(result, Response):
                return result
        handler, path_params = self._match_route(request.method, request.path)
        if not handler:
            return Response.error(f"Not found: {request.method} {request.path}", 404)
        request.path_params = path_params
        try:
            result = handler(request)
            return result if isinstance(result, Response) else Response.json(result)
        except Exception as e:
            return Response.error(str(e), 500)
'''

    def _generate_user_service(self, context: GenerationContext) -> str:
        """Generate a working user service."""
        return '''"""
UserService - User management with validation.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import re
import hashlib


@dataclass
class User:
    id: str
    username: str
    email: str
    password_hash: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class UserService:
    """User management service with validation."""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.email_index: Dict[str, str] = {}
    
    def _hash_password(self, password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _validate_email(self, email: str) -> bool:
        return bool(re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$', email))
    
    def create_user(self, username: str, email: str, password: str) -> User:
        if not self._validate_email(email):
            raise ValueError(f"Invalid email: {email}")
        if email in self.email_index:
            raise ValueError(f"Email exists: {email}")
        user_id = str(uuid.uuid4())
        user = User(id=user_id, username=username, email=email, password_hash=self._hash_password(password))
        self.users[user_id] = user
        self.email_index[email] = user_id
        return user
    
    def get_user(self, user_id: str) -> Optional[User]:
        return self.users.get(user_id)
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        user_id = self.email_index.get(email)
        return self.users.get(user_id) if user_id else None
    
    def update_user(self, user_id: str, data: Dict[str, Any]) -> User:
        user = self.users.get(user_id)
        if not user:
            raise ValueError(f"User not found: {user_id}")
        if 'username' in data:
            user.username = data['username']
        if 'email' in data:
            new_email = data['email']
            if not self._validate_email(new_email):
                raise ValueError(f"Invalid email: {new_email}")
            del self.email_index[user.email]
            self.email_index[new_email] = user_id
            user.email = new_email
        if 'password' in data:
            user.password_hash = self._hash_password(data['password'])
        return user
    
    def delete_user(self, user_id: str) -> bool:
        user = self.users.get(user_id)
        if not user:
            return False
        del self.email_index[user.email]
        del self.users[user_id]
        return True
    
    def list_users(self, limit: int = 100, offset: int = 0) -> List[User]:
        return list(self.users.values())[offset:offset + limit]
    
    def authenticate(self, email: str, password: str) -> Optional[User]:
        user = self.get_user_by_email(email)
        if user and user.password_hash == self._hash_password(password):
            return user
        return None
'''

    def _generate_template_renderer(self, context: GenerationContext) -> str:
        """Generate a working template renderer."""
        return '''"""
TemplateRenderer - HTML template rendering with variable substitution.
"""

from typing import Dict, Any
import re
import html


class TemplateRenderer:
    """HTML template renderer for frontend."""
    
    def __init__(self, template_dir: str = "templates"):
        self.template_dir = template_dir
        self.cache: Dict[str, str] = {}
    
    def escape_html(self, value: str) -> str:
        return html.escape(str(value))
    
    def load_template(self, template_name: str) -> str:
        if template_name in self.cache:
            return self.cache[template_name]
        try:
            with open(f"{self.template_dir}/{template_name}", 'r') as f:
                template = f.read()
            self.cache[template_name] = template
            return template
        except FileNotFoundError:
            raise ValueError(f"Template not found: {template_name}")
    
    def render(self, template_name: str, context: Dict[str, Any]) -> str:
        template = self.load_template(template_name)
        return self.render_string(template, context)
    
    def render_string(self, template: str, context: Dict[str, Any]) -> str:
        result = template
        def replace_var(match):
            var_name = match.group(1).strip()
            if var_name not in context:
                raise ValueError(f"Missing variable: {var_name}")
            return self.escape_html(context[var_name])
        result = re.sub(r'\\{\\{\\s*(\\w+)\\s*\\}\\}', replace_var, result)
        def replace_raw(match):
            var_name = match.group(1).strip()
            if var_name not in context:
                raise ValueError(f"Missing variable: {var_name}")
            return str(context[var_name])
        result = re.sub(r'\\{\\{\\{\\s*(\\w+)\\s*\\}\\}\\}', replace_raw, result)
        return result
    
    def add_template(self, name: str, content: str) -> None:
        self.cache[name] = content
'''

    def _generate_static_server(self, context: GenerationContext) -> str:
        """Generate a working static file server."""
        return '''"""
StaticFileServer - Serve static files with proper MIME types.
"""

from typing import Dict
from dataclasses import dataclass, field
import os
import mimetypes


@dataclass
class Response:
    status_code: int
    headers: Dict[str, str] = field(default_factory=dict)
    body: bytes = b''
    
    @staticmethod
    def error(message: str, status: int = 400) -> 'Response':
        return Response(status, {'Content-Type': 'text/plain'}, message.encode())


class StaticFileServer:
    """Static file server for frontend assets."""
    
    def __init__(self, static_dir: str = "static"):
        self.static_dir = os.path.abspath(static_dir)
        self.mime_types = {
            '.html': 'text/html', '.css': 'text/css', '.js': 'application/javascript',
            '.json': 'application/json', '.png': 'image/png', '.jpg': 'image/jpeg',
            '.gif': 'image/gif', '.svg': 'image/svg+xml', '.ico': 'image/x-icon',
        }
        self.cache_max_age = 3600
    
    def get_mime_type(self, path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        if ext in self.mime_types:
            return self.mime_types[ext]
        mime, _ = mimetypes.guess_type(path)
        return mime or 'application/octet-stream'
    
    def serve(self, path: str) -> Response:
        path = path.lstrip('/')
        full_path = os.path.abspath(os.path.join(self.static_dir, path))
        if not full_path.startswith(self.static_dir):
            return Response.error("Forbidden", 403)
        if not os.path.isfile(full_path):
            return Response.error(f"Not found: {path}", 404)
        try:
            with open(full_path, 'rb') as f:
                content = f.read()
            return Response(200, {
                'Content-Type': self.get_mime_type(path),
                'Content-Length': str(len(content)),
                'Cache-Control': f'max-age={self.cache_max_age}',
            }, content)
        except IOError as e:
            return Response.error(f"Error: {e}", 500)
'''

    def _generate_webapp(self, context: GenerationContext) -> str:
        """Generate a working web application."""
        return '''"""
WebApp - Main web application orchestrator.
"""

from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
import json
import socket


@dataclass
class Request:
    method: str
    path: str
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[Dict[str, Any]] = None
    path_params: Dict[str, str] = field(default_factory=dict)


@dataclass
class Response:
    status_code: int
    headers: Dict[str, str] = field(default_factory=dict)
    body: Any = None
    
    @staticmethod
    def json(data: Any, status: int = 200) -> 'Response':
        return Response(status, {'Content-Type': 'application/json'}, data)
    
    @staticmethod
    def html(content: str, status: int = 200) -> 'Response':
        return Response(status, {'Content-Type': 'text/html'}, content)
    
    @staticmethod
    def error(message: str, status: int = 400) -> 'Response':
        return Response.json({'error': message}, status)


class WebApp:
    """Main web application orchestrator."""
    
    def __init__(self, name: str = "WebApp"):
        self.name = name
        self.routes: Dict[str, Dict[str, Callable]] = {'GET': {}, 'POST': {}, 'PUT': {}, 'DELETE': {}}
        self._running = False
        self._socket = None
    
    def get(self, path: str) -> Callable:
        def decorator(handler: Callable) -> Callable:
            self.routes['GET'][path] = handler
            return handler
        return decorator
    
    def post(self, path: str) -> Callable:
        def decorator(handler: Callable) -> Callable:
            self.routes['POST'][path] = handler
            return handler
        return decorator
    
    def handle_request(self, request: Request) -> Response:
        routes = self.routes.get(request.method, {})
        if request.path in routes:
            try:
                result = routes[request.path](request)
                if isinstance(result, Response):
                    return result
                if isinstance(result, dict):
                    return Response.json(result)
                return Response.html(str(result))
            except Exception as e:
                return Response.error(str(e), 500)
        return Response.error(f"Not found: {request.path}", 404)
    
    def _parse_request(self, data: bytes) -> Optional[Request]:
        try:
            text = data.decode('utf-8')
            lines = text.split('\\r\\n')
            method, path, _ = lines[0].split(' ')
            headers = {}
            i = 1
            while i < len(lines) and lines[i]:
                if ':' in lines[i]:
                    key, value = lines[i].split(':', 1)
                    headers[key.strip()] = value.strip()
                i += 1
            body = None
            if i < len(lines) - 1:
                body_text = '\\r\\n'.join(lines[i+1:])
                if body_text.strip():
                    try:
                        body = json.loads(body_text)
                    except:
                        body = {'raw': body_text}
            return Request(method=method, path=path, headers=headers, body=body)
        except:
            return None
    
    def _format_response(self, response: Response) -> bytes:
        status_msgs = {200: 'OK', 201: 'Created', 400: 'Bad Request', 404: 'Not Found', 500: 'Internal Server Error'}
        if isinstance(response.body, (dict, list)):
            body = json.dumps(response.body).encode()
        elif isinstance(response.body, str):
            body = response.body.encode()
        elif isinstance(response.body, bytes):
            body = response.body
        else:
            body = b''
        lines = [f"HTTP/1.1 {response.status_code} {status_msgs.get(response.status_code, 'Unknown')}"]
        for key, value in response.headers.items():
            lines.append(f"{key}: {value}")
        lines.append(f"Content-Length: {len(body)}")
        lines.append("")
        return '\\r\\n'.join(lines).encode() + b'\\r\\n' + body
    
    def start(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((host, port))
        self._socket.listen(5)
        self._running = True
        print(f"[{self.name}] Server running at http://{host}:{port}")
        while self._running:
            try:
                self._socket.settimeout(1.0)
                try:
                    client, _ = self._socket.accept()
                except socket.timeout:
                    continue
                data = client.recv(4096)
                request = self._parse_request(data)
                if request:
                    response = self.handle_request(request)
                    client.send(self._format_response(response))
                client.close()
            except Exception as e:
                if self._running:
                    print(f"Error: {e}")
    
    def stop(self) -> None:
        self._running = False
        if self._socket:
            self._socket.close()
'''

    def _generate_intent_classifier(self, context: GenerationContext) -> str:
        """Generate a working intent classifier."""
        return '''"""
IntentClassifier - Simple keyword-based intent classification.
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field
import re


@dataclass
class Intent:
    name: str
    confidence: float
    entities: Dict[str, Any] = field(default_factory=dict)


class IntentClassifier:
    """Classifies user messages into intents."""
    
    def __init__(self):
        self.patterns: Dict[str, List[str]] = {
            'greeting': ['hello', 'hi', 'hey', 'good morning', 'howdy'],
            'farewell': ['bye', 'goodbye', 'see you', 'later', 'take care'],
            'help': ['help', 'assist', 'how do', 'what is', 'explain'],
            'thanks': ['thank', 'thanks', 'appreciate'],
            'affirmative': ['yes', 'yeah', 'sure', 'ok', 'correct'],
            'negative': ['no', 'nope', 'not', 'never', 'wrong'],
        }
        self.entity_patterns = {
            'email': r'[\\w.+-]+@[\\w-]+\\.[\\w.-]+',
            'number': r'\\b\\d+\\b',
        }
        self.threshold = 0.3
    
    def classify(self, message: str) -> Intent:
        message_lower = message.lower()
        scores = {}
        for intent, keywords in self.patterns.items():
            score = sum(len(kw)/10.0 for kw in keywords if kw in message_lower)
            scores[intent] = min(score, 1.0)
        if scores:
            best = max(scores, key=scores.get)
            if scores[best] >= self.threshold:
                return Intent(best, scores[best], self.extract_entities(message))
        return Intent('unknown', 0.0)
    
    def extract_entities(self, message: str) -> Dict[str, Any]:
        entities = {}
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, message)
            if matches:
                entities[entity_type] = matches[0] if len(matches) == 1 else matches
        return entities
    
    def add_intent(self, name: str, keywords: List[str]) -> None:
        self.patterns[name] = keywords
'''

    def _generate_response_generator(self, context: GenerationContext) -> str:
        """Generate a working response generator."""
        return '''"""
ResponseGenerator - Template-based response generation.
"""

from typing import Dict, List, Any
import random


class ResponseGenerator:
    """Generates responses based on intent and context."""
    
    def __init__(self):
        self.templates: Dict[str, List[str]] = {
            'greeting': ["Hello! How can I help?", "Hi there! What can I do for you?"],
            'farewell': ["Goodbye! Have a great day!", "See you later!"],
            'help': ["I'm here to help! What do you need?", "Sure, what's your question?"],
            'thanks': ["You're welcome!", "Happy to help!"],
            'affirmative': ["Great! Let's proceed.", "Perfect!"],
            'negative': ["No problem. What would you like instead?", "Understood."],
            'unknown': ["I'm not sure I understand. Could you rephrase?", "Can you try again?"],
        }
        self._context: Dict[str, Any] = {}
    
    def generate(self, intent, conversation=None) -> str:
        templates = self.templates.get(intent.name, self.templates['unknown'])
        last = self._context.get('last_response', '')
        available = [t for t in templates if t != last] or templates
        response = random.choice(available)
        for key, value in intent.entities.items():
            response = response.replace(f'{{{key}}}', str(value))
        self._context['last_response'] = response
        return response
    
    def set_template(self, intent_name: str, templates: List[str]) -> None:
        self.templates[intent_name] = templates
'''

    def _generate_conversation_manager(self, context: GenerationContext) -> str:
        """Generate a working conversation manager."""
        return '''"""
ConversationManager - Manage conversation state and history.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid


@dataclass
class Message:
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Conversation:
    id: str
    messages: List[Message] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, role: str, content: str) -> Message:
        msg = Message(role=role, content=content)
        self.messages.append(msg)
        return msg
    
    def get_history(self, limit: int = 10) -> List[Message]:
        return self.messages[-limit:]


class ConversationManager:
    """Manages conversation state and history."""
    
    def __init__(self, max_history: int = 100):
        self.conversations: Dict[str, Conversation] = {}
        self.max_history = max_history
    
    def create_conversation(self, user_id: Optional[str] = None) -> Conversation:
        conv_id = str(uuid.uuid4())
        conv = Conversation(id=conv_id)
        if user_id:
            conv.context['user_id'] = user_id
        self.conversations[conv_id] = conv
        return conv
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        return self.conversations.get(conversation_id)
    
    def add_message(self, conversation_id: str, role: str, content: str) -> Message:
        conv = self.conversations.get(conversation_id)
        if not conv:
            raise ValueError(f"Conversation not found: {conversation_id}")
        msg = conv.add_message(role, content)
        if len(conv.messages) > self.max_history:
            conv.messages = conv.messages[-self.max_history:]
        return msg
    
    def delete_conversation(self, conversation_id: str) -> bool:
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            return True
        return False
'''

    def _generate_chatbot(self, context: GenerationContext) -> str:
        """Generate a working chatbot."""
        return '''"""
ChatBot - Main chatbot orchestrator.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import random


@dataclass
class Message:
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Intent:
    name: str
    confidence: float
    entities: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Conversation:
    id: str
    messages: List[Message] = field(default_factory=list)


class ChatBot:
    """Main chatbot orchestrator."""
    
    def __init__(self, name: str = "Assistant"):
        self.name = name
        self.conversations: Dict[str, Conversation] = {}
        self.templates: Dict[str, List[str]] = {
            'greeting': [f"Hello! I'm {name}. How can I help?"],
            'farewell': ["Goodbye! Have a great day!"],
            'help': ["I can help with general questions. Just ask!"],
            'unknown': ["I'm not sure I understand. Could you rephrase?"],
        }
        self.patterns: Dict[str, List[str]] = {
            'greeting': ['hello', 'hi', 'hey'],
            'farewell': ['bye', 'goodbye', 'see you'],
            'help': ['help', 'assist', 'how'],
        }
    
    def start_conversation(self) -> str:
        conv_id = str(uuid.uuid4())
        self.conversations[conv_id] = Conversation(id=conv_id)
        return conv_id
    
    def end_conversation(self, conversation_id: str) -> None:
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
    
    def get_history(self, conversation_id: str) -> List[Message]:
        conv = self.conversations.get(conversation_id)
        return conv.messages if conv else []
    
    def chat(self, message: str, conversation_id: Optional[str] = None) -> str:
        if not conversation_id or conversation_id not in self.conversations:
            conversation_id = self.start_conversation()
        conv = self.conversations[conversation_id]
        conv.messages.append(Message(role='user', content=message))
        intent = self._classify(message)
        response = self._generate_response(intent)
        conv.messages.append(Message(role='assistant', content=response))
        return response
    
    def _classify(self, message: str) -> Intent:
        message_lower = message.lower()
        for intent_name, keywords in self.patterns.items():
            if any(kw in message_lower for kw in keywords):
                return Intent(intent_name, 0.8)
        return Intent('unknown', 0.0)
    
    def _generate_response(self, intent: Intent) -> str:
        templates = self.templates.get(intent.name, self.templates['unknown'])
        return random.choice(templates)
    
    def add_intent(self, name: str, keywords: List[str], responses: List[str]) -> None:
        self.patterns[name] = keywords
        self.templates[name] = responses
    
    def run_cli(self) -> None:
        print(f"\\n{self.name} is ready! Type 'quit' to exit.\\n")
        conv_id = self.start_conversation()
        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print(f"{self.name}: Goodbye!")
                    break
                if user_input:
                    response = self.chat(user_input, conv_id)
                    print(f"{self.name}: {response}")
            except KeyboardInterrupt:
                print(f"\\n{self.name}: Goodbye!")
                break
        self.end_conversation(conv_id)
'''

    def _generate_generic(self, context: GenerationContext) -> str:
        """Generate a generic working implementation."""
        protocol = context.protocol
        lines = [
            f'"""',
            f'{protocol.name} implementation.',
            f'{protocol.docstring}',
            f'"""',
            f'',
            f'from typing import Dict, List, Any, Optional',
            f'import uuid',
            f'',
            f'',
            f'class {protocol.name}:',
            f'    """',
            f'    {protocol.docstring}',
        ]
        if protocol.pac_invariants:
            lines.append(f'    ')
            lines.append(f'    PAC Invariants:')
            for inv in protocol.pac_invariants:
                lines.append(f'        - {inv}')
        lines.extend([
            f'    """',
            f'    ',
            f'    def __init__(self):',
            f'        """Initialize {protocol.name}."""',
            f'        self._data: Dict[str, Any] = {{}}',
            f'    ',
        ])
        for method in protocol.methods:
            lines.extend(self._generate_working_method(method))
            lines.append(f'    ')
        return '\n'.join(lines)
    
    def _generate_working_method(self, method: str) -> List[str]:
        """Generate a working method implementation."""
        if method in ['get', 'fetch', 'retrieve', 'read']:
            return [
                f'    def {method}(self, key: str) -> Any:',
                f'        """Retrieve item by key."""',
                f'        return self._data.get(key)',
            ]
        elif method in ['post', 'create', 'add', 'insert']:
            return [
                f'    def {method}(self, data: Dict[str, Any]) -> Dict[str, Any]:',
                f'        """Create new item."""',
                f'        item_id = str(uuid.uuid4())',
                f'        self._data[item_id] = data',
                f'        return {{"id": item_id, **data}}',
            ]
        elif method in ['put', 'update', 'modify']:
            return [
                f'    def {method}(self, key: str, data: Dict[str, Any]) -> Dict[str, Any]:',
                f'        """Update existing item."""',
                f'        if key not in self._data:',
                f'            raise KeyError(f"Not found: {{key}}")',
                f'        self._data[key].update(data)',
                f'        return self._data[key]',
            ]
        elif method in ['delete', 'remove']:
            return [
                f'    def {method}(self, key: str) -> bool:',
                f'        """Delete item by key."""',
                f'        if key in self._data:',
                f'            del self._data[key]',
                f'            return True',
                f'        return False',
            ]
        elif method in ['list', 'list_all', 'get_all']:
            return [
                f'    def {method}(self, limit: int = 100, offset: int = 0) -> List[Any]:',
                f'        """List all items with pagination."""',
                f'        return list(self._data.values())[offset:offset + limit]',
            ]
        else:
            return [
                f'    def {method}(self, *args, **kwargs) -> Any:',
                f'        """Execute {method}."""',
                f'        return {{"method": "{method}", "args": args, "kwargs": kwargs}}',
            ]


class RandomVariationGenerator(MockGenerator):
    """Mock generator with random variations for testing evolution."""
    
    def __init__(self, variation_rate: float = 0.3):
        super().__init__()
        self.variation_rate = variation_rate
    
    @property
    def name(self) -> str:
        return "mock-random"
