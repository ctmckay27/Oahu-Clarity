FROM mcr.microsoft.com/playwright:v1.56.1-noble
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends xvfb x11vnc novnc websockify x11-utils ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY browser-actuator/package.json ./package.json
RUN npm install --omit=dev
COPY browser-actuator/server.js ./server.js
COPY browser-actuator/start.sh ./start.sh
RUN chmod +x /app/start.sh
ENV DISPLAY=:99 PROFILE_DIR=/tmp/mari404-chrome-profile
EXPOSE 10000
CMD ["/app/start.sh"]
