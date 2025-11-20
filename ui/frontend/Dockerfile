FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package.json ./

# Clear npm cache and install dependencies
RUN npm cache clean --force && \
    npm install --legacy-peer-deps && \
    npm install ajv@^8.0.0 ajv-keywords@^5.0.0 --legacy-peer-deps

# Copy project files
COPY . .

# Expose port
EXPOSE 3000

# Start development server
CMD ["npm", "start"]
