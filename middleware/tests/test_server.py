import pytest
import asyncio
import websockets
import json
from unittest.mock import AsyncMock, patch
from middleware.server import MinecraftWebSocketServer

@pytest.fixture
def server():
    return MinecraftWebSocketServer()

@pytest.mark.asyncio
async def test_register_unregister_client(server):
    """Test client registration and unregistration"""
    # Mock websocket
    mock_websocket = AsyncMock()
    
    # Test registration
    await server.register_client(mock_websocket)
    assert mock_websocket in server.connected_clients
    assert len(server.connected_clients) == 1
    
    # Test unregistration
    await server.unregister_client(mock_websocket)
    assert mock_websocket not in server.connected_clients
    assert len(server.connected_clients) == 0

@pytest.mark.asyncio
async def test_handle_student_join(server):
    """Test student join message handling"""
    mock_websocket = AsyncMock()
    
    join_data = {
        'type': 'student_join',
        'user_id': 'test_user_123',
        'username': 'TestStudent'
    }
    
    with patch.object(server.tracker, 'log_session_start') as mock_log:
        await server.handle_student_join(mock_websocket, join_data)
        
        # Check if session was registered
        assert 'test_user_123' in server.student_sessions
        assert server.student_sessions['test_user_123']['username'] == 'TestStudent'
        
        # Check if tracker was called
        mock_log.assert_called_once_with('test_user_123', 'TestStudent')
        
        # Check if welcome message was sent
        mock_websocket.send.assert_called_once()

@pytest.mark.asyncio
async def test_handle_chat_request(server):
    """Test chat request handling"""
    mock_websocket = AsyncMock()
    
    # Setup student session
    server.student_sessions['test_user'] = {
        'websocket': mock_websocket,
        'username': 'TestUser'
    }
    
    chat_data = {
        'type': 'chat_request',
        'user_id': 'test_user',
        'message': 'Co to jest matematyka?'
    }
    
    # Mock backend response
    mock_response = {
        'response': 'Matematyka to nauka o liczbach i wzorach.',
        'sources': []
    }
    
    with patch('aiohttp.ClientSession') as mock_session:
        # Configure mock HTTP response
        mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value.status = 200
        mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value.json.return_value = mock_response
        
        with patch.object(server.tracker, 'log_chat_interaction') as mock_log:
            await server.handle_chat_request(mock_websocket, chat_data)
            
            # Check if response was sent
            mock_websocket.send.assert_called()
            
            # Check if interaction was logged
            mock_log.assert_called_once()

@pytest.mark.asyncio
async def test_invalid_json_handling(server):
    """Test handling of invalid JSON messages"""
    mock_websocket = AsyncMock()
    
    await server.handle_minecraft_message(mock_websocket, "invalid json")
    
    # Check if error message was sent
    mock_websocket.send.assert_called_once()
    
    # Verify error response format
    call_args = mock_websocket.send.call_args[0][0]
    error_response = json.loads(call_args)
    assert error_response['type'] == 'error'
    assert 'Invalid JSON format' in error_response['message']