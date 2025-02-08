#!/bin/bash

# Function to display help message
show_help() {
    echo "Usage: $0 [command]"
    echo
    echo "Commands:"
    echo "  start       Start all containers"
    echo "  stop        Stop all containers"
    echo "  restart     Restart all containers"
    echo "  build       Build all containers"
    echo "  logs        Show container logs"
    echo "  status      Show container status"
    echo "  clean       Remove all containers and volumes"
    echo "  help        Show this help message"
}

# Function to start containers
start() {
    echo "Starting containers..."
    docker-compose up -d
}

# Function to stop containers
stop() {
    echo "Stopping containers..."
    docker-compose down
}

# Function to restart containers
restart() {
    echo "Restarting containers..."
    docker-compose restart
}

# Function to build containers
build() {
    echo "Building containers..."
    docker-compose build --no-cache
}

# Function to show logs
logs() {
    echo "Showing logs..."
    docker-compose logs -f
}

# Function to show status
status() {
    echo "Container status:"
    docker-compose ps
}

# Function to clean up
clean() {
    echo "Cleaning up..."
    docker-compose down -v
    docker system prune -f
}

# Main script
case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    build)
        build
        ;;
    logs)
        logs
        ;;
    status)
        status
        ;;
    clean)
        clean
        ;;
    help|*)
        show_help
        ;;
esac