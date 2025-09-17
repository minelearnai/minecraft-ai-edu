# Minecraft AI Edu 🎮🤖

AI-powered educational platform integrating with Minecraft Education Edition for interactive STEM learning.

## Project Overview

This project combines AI tutoring with Minecraft Education Edition to create an engaging learning experience for students aged 10-15, focusing on mathematics and physics education.

## Features

- 🤖 AI-powered chatbot for educational support
- 🎮 Minecraft Education Edition integration via WebSocket
- 📊 Student activity tracking and analytics
- 🎯 Personalized learning experiences
- 👨‍🏫 Teacher dashboard for monitoring progress

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Minecraft Education Edition
- Python 3.11+

### Running Locally

1. Clone the repository:
```bash
git clone https://github.com/minelearnai/minecraft-ai-edu.git
cd minecraft-ai-edu
```

2. Start the services:
```bash
docker-compose up --build
```

3. Initialize the knowledge base:
```bash
docker-compose up --build etl
```

4. Access the services:
- Backend API: http://localhost:8000
- WebSocket Server: ws://localhost:3000
- Elasticsearch: http://localhost:9200
- MongoDB: localhost:27017

### Minecraft Integration

In Minecraft Education Edition, connect to the WebSocket server:
```
/connect localhost:3000
```

## Architecture

- **Backend**: FastAPI with AI provider abstraction (OpenAI/Llama2)
- **Middleware**: WebSocket server for Minecraft communication
- **Knowledge Base**: Elasticsearch with educational content
- **Tracking**: MongoDB for student activity analytics
- **Frontend**: React.js teacher dashboard (coming soon)

## Development

### Project Structure
```
minecraft-ai-edu/
├── backend/           # FastAPI AI chatbot service
├── middleware/        # WebSocket server for Minecraft
├── knowledge/         # Knowledge base and ETL scripts
├── tracking/          # Student activity tracking
├── frontend/          # Teacher dashboard (React)
└── docs/              # Documentation
```

### Environment Variables

- `USE_LOCAL_LM=1` - Use local Llama2 instead of OpenAI
- `OPENAI_API_KEY` - OpenAI API key for GPT-4
- `ELASTICSEARCH_URL=http://elasticsearch:9200`
- `MONGODB_URL=mongodb://mongo:27017`

## Educational Focus

### Target Audience
- Students: Grades 4-8 (ages 10-15)
- Subjects: Mathematics, Physics, Programming Logic
- Proven effectiveness through research-backed methodologies

### Sample Learning Scenarios
- Building geometric shapes to learn area/volume calculations
- Creating circuits with Redstone to understand electricity
- Programming logic through command blocks

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Research & References

This project is based on extensive research showing game-based learning effectiveness, particularly:
- Queensland University study showing 80% confidence boost in mathematics
- Harvard research on AI tutoring being 2x more effective
- Meta-analysis confirming optimal results for 10-15 age group

## Team

- **Project Manager**: Universal coordinator and business development
- **Technical Lead**: Full-stack development and AI integration  
- **Educational Specialist**: Curriculum design and pedagogical approach

---

**Status**: MVP Development Phase (Sept 2025 - Nov 2025)