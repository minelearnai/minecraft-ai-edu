import asyncio
import websockets
import json
import aiohttp
import os
from datetime import datetime
from typing import Dict, Set
from .tracking import StudentTracker

class MinecraftWebSocketServer:
    def __init__(self):
        self.connected_clients: Set[websockets.WebSocketServerProtocol] = set()
        self.student_sessions: Dict[str, dict] = {}
        self.backend_url = os.getenv('BACKEND_URL', 'http://localhost:8000')
        self.tracker = StudentTracker()
    
    async def register_client(self, websocket: websockets.WebSocketServerProtocol):
        """Register a new WebSocket client"""
        self.connected_clients.add(websocket)
        print(f"🔗 New client connected. Total: {len(self.connected_clients)}")
    
    async def unregister_client(self, websocket: websockets.WebSocketServerProtocol):
        """Unregister a WebSocket client"""
        self.connected_clients.discard(websocket)
        
        # Clean up student session if exists
        user_id = None
        for uid, session in self.student_sessions.items():
            if session.get('websocket') == websocket:
                user_id = uid
                break
        
        if user_id:
            await self.tracker.log_session_end(user_id)
            del self.student_sessions[user_id]
            print(f"📤 Student {user_id} disconnected")
        
        print(f"❌ Client disconnected. Total: {len(self.connected_clients)}")
    
    async def handle_minecraft_message(self, websocket: websockets.WebSocketServerProtocol, message: str):
        """Handle incoming message from Minecraft Education Edition"""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type == 'student_join':
                await self.handle_student_join(websocket, data)
            elif message_type == 'chat_request':
                await self.handle_chat_request(websocket, data)
            elif message_type == 'activity':
                await self.handle_student_activity(websocket, data)
            elif message_type == 'heartbeat':
                await websocket.send(json.dumps({'type': 'heartbeat_response', 'timestamp': datetime.now().isoformat()}))
            else:
                print(f"⚠️ Unknown message type: {message_type}")
        
        except json.JSONDecodeError:
            await websocket.send(json.dumps({
                'type': 'error', 
                'message': 'Invalid JSON format'
            }))
        except Exception as e:
            print(f"❌ Error handling message: {e}")
            await websocket.send(json.dumps({
                'type': 'error', 
                'message': f'Server error: {str(e)}'
            }))
    
    async def handle_student_join(self, websocket: websockets.WebSocketServerProtocol, data: dict):
        """Handle student joining the session"""
        user_id = data.get('user_id')
        username = data.get('username', 'Unknown')
        
        if not user_id:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': 'user_id is required'
            }))
            return
        
        # Register student session
        self.student_sessions[user_id] = {
            'websocket': websocket,
            'username': username,
            'joined_at': datetime.now(),
            'last_activity': datetime.now()
        }
        
        # Log session start
        await self.tracker.log_session_start(user_id, username)
        
        # Send welcome message
        await websocket.send(json.dumps({
            'type': 'welcome',
            'message': f'Witaj {username}! Jestem Twoim asystentem AI. Jak mogę Ci pomóc w nauce?',
            'user_id': user_id
        }))
        
        print(f"🎓 Student {username} ({user_id}) joined the session")
    
    async def handle_chat_request(self, websocket: websockets.WebSocketServerProtocol, data: dict):
        """Handle chat request from student"""
        user_id = data.get('user_id')
        message = data.get('message')
        
        if not user_id or not message:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': 'user_id and message are required'
            }))
            return
        
        # Update last activity
        if user_id in self.student_sessions:
            self.student_sessions[user_id]['last_activity'] = datetime.now()
        
        try:
            # Forward to backend AI service
            async with aiohttp.ClientSession() as session:
                payload = {
                    'user_id': user_id,
                    'message': message,
                    'context': data.get('context')
                }
                
                async with session.post(f'{self.backend_url}/chat', json=payload) as response:
                    if response.status == 200:
                        ai_response = await response.json()
                        
                        # Log the interaction
                        await self.tracker.log_chat_interaction(user_id, message, ai_response['response'])
                        
                        # Send response back to Minecraft
                        await websocket.send(json.dumps({
                            'type': 'chat_response',
                            'response': ai_response['response'],
                            'sources': ai_response.get('sources', []),
                            'user_id': user_id
                        }))
                    else:
                        error_msg = f'Backend error: {response.status}'
                        await websocket.send(json.dumps({
                            'type': 'error',
                            'message': error_msg
                        }))
        
        except Exception as e:
            print(f"❌ Error forwarding chat request: {e}")
            await websocket.send(json.dumps({
                'type': 'error',
                'message': 'Failed to get AI response'
            }))
    
    async def handle_student_activity(self, websocket: websockets.WebSocketServerProtocol, data: dict):
        """Handle student activity tracking"""
        user_id = data.get('user_id')
        activity_type = data.get('activity_type')
        details = data.get('details', {})
        
        if user_id and activity_type:
            await self.tracker.log_activity(user_id, activity_type, details)
            
            # Update session activity
            if user_id in self.student_sessions:
                self.student_sessions[user_id]['last_activity'] = datetime.now()
    
    async def client_handler(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Handle individual WebSocket client connection"""
        await self.register_client(websocket)
        
        try:
            async for message in websocket:
                await self.handle_minecraft_message(websocket, message)
        except websockets.exceptions.ConnectionClosedOK:
            pass
        except websockets.exceptions.ConnectionClosedError:
            pass
        except Exception as e:
            print(f"❌ Client handler error: {e}")
        finally:
            await self.unregister_client(websocket)
    
    async def start_server(self, host='0.0.0.0', port=3000):
        """Start the WebSocket server"""
        print(f"🚀 Starting Minecraft WebSocket server on {host}:{port}")
        
        async with websockets.serve(self.client_handler, host, port):
            print(f"✅ WebSocket server running on ws://{host}:{port}")
            await asyncio.Future()  # Run forever

# Main entry point
if __name__ == "__main__":
    server = MinecraftWebSocketServer()
    asyncio.run(server.start_server())