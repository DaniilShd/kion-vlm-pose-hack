#!/bin/bash

echo "🚀 Запускаем контейнер..."
docker-compose up -d
echo "✅ Контейнер запущен"

echo "📝 Проверяем статус:"
docker-compose ps

echo -e "\n💻 Для входа в контейнер выполните:"
echo "  docker exec -it kion-moderator bash"

