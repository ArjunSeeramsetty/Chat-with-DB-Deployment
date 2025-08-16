#!/bin/sh
#
# This script is used to inject runtime environment variables into the frontend's index.html.
# This allows the same Docker image to be used across different environments (dev, staging, prod)
# by just changing the environment variables passed to the container.

# The root directory of the nginx server
ROOT_DIR=/usr/share/nginx/html

# Create a config.js file
CONFIG_FILE=$ROOT_DIR/config.js
echo "window.API_BASE_URL = '${API_BASE_URL:-http://localhost:8000}';" > $CONFIG_FILE

# Add the config.js script to index.html
# The sed command will add the script tag before the closing </head> tag.
sed -i '/<\/head>/i <script src="/config.js"></script>' $ROOT_DIR/index.html

# Start nginx
exec nginx -g 'daemon off;'
