let symbols = [
  { id: 1, x: 200, y: 150, strength: 1.0, width: 30 },
  { id: 2, x: 350, y: 200, strength: 0.8, width: 40 },
  { id: 3, x: 150, y: 300, strength: 0.7, width: 35 }
];

// Parameters for energy function
const repulsionStrength = 0.3;
const repulsionRange = 100;

// Canvas dimensions
const canvasWidth = 800;
const canvasHeight = 600;
const resolution = 5;

function setup() {
  const canvas = createCanvas(canvasWidth, canvasHeight);
  canvas.parent('canvasContainer');
}

function draw() {
  background(240);
  
  // Generate grid of points
  const points = [];
  for (let x = 0; x < canvasWidth; x += resolution) {
    for (let y = 0; y < height; y += resolution) {
      points.push({ x, y });
    }
  }
  
  // Calculate energy for each point
  points.forEach(point => {
    let energy = 0;
    
    // Attraction to symbols (negative energy)
    symbols.forEach(symbol => {
      const dist = Math.sqrt(Math.pow(point.x - symbol.x, 2) + Math.pow(point.y - symbol.y, 2));
      energy -= symbol.strength * Math.exp(-Math.pow(dist, 2) / Math.pow(symbol.width, 2));
    });
    
    point.energy = energy;
  });
  
  // Find energy range for color scaling
  let minEnergy = Infinity;
  let maxEnergy = -Infinity;
  points.forEach(point => {
    minEnergy = Math.min(minEnergy, point.energy);
    maxEnergy = Math.max(maxEnergy, point.energy);
  });
  
  // Draw energy landscape
  noStroke();
  points.forEach(point => {
    const normalizedEnergy = map(point.energy, minEnergy, maxEnergy, 0, 1);
    const color = getViridisColor(normalizedEnergy);
    fill(color.r, color.g, color.b);
    rect(point.x, point.y, resolution, resolution);
  });
  
  // Draw symbols
  symbols.forEach(symbol => {
    fill(255, 0, 0);
    stroke(255);
    strokeWeight(1);
    circle(symbol.x, symbol.y, 5);
  });
  
  // Draw legend
  drawLegend(minEnergy, maxEnergy);
  
  // Only draw once if there are no interactions
  if (frameCount > 1 && mouseIsPressed === false) {
    noLoop();
  }
}

window.handleAddSymbol = function() {
  // Add a random new symbol
  const newSymbol = {
    id: symbols.length + 1,
    x: random(50, canvasWidth - 50),
    y: random(50, canvasHeight - 50),
    strength: random(0.5, 1.0),
    width: random(25, 50)
  };
  symbols.push(newSymbol);
  loop(); // Restart drawing to update visualization
};

window.handleClear = function() {
  symbols = [];
  loop(); // Restart drawing to update visualization
};

function drawLegend(minEnergy, maxEnergy) {
  const legendWidth = 20;
  const legendHeight = 200;
  const legendX = canvasWidth - 50;
  const legendY = 50;
  
  // Draw legend gradient
  for (let i = 0; i < legendHeight; i++) {
    const normalizedValue = i / legendHeight;
    const color = getViridisColor(1 - normalizedValue);
    fill(color.r, color.g, color.b);
    noStroke();
    rect(legendX, legendY + i, legendWidth, 1);
  }
  
  // Draw legend axis
  stroke(0);
  strokeWeight(1);
  line(legendX + legendWidth, legendY, legendX + legendWidth, legendY + legendHeight);
  
  // Draw ticks and labels
  const numTicks = 5;
  textAlign(LEFT, CENTER);
  textSize(10);
  for (let i = 0; i <= numTicks; i++) {
    const y = legendY + (i / numTicks) * legendHeight;
    const value = map(i / numTicks, 0, 1, minEnergy, maxEnergy);
    line(legendX + legendWidth, y, legendX + legendWidth + 5, y);
    text(value.toFixed(2), legendX + legendWidth + 8, y);
  }
  
  // Draw label
  push();
  translate(legendX - 10, legendY + legendHeight / 2);
  rotate(-HALF_PI);
  textAlign(CENTER, CENTER);
  text("Energy", 0, 0);
  pop();
}

// Function to get a color from the Viridis color scheme
function getViridisColor(t) {
  // Approximation of Viridis colormap
  let r, g, b;
  
  if (t <= 0.0) {
    r = 68; g = 1; b = 84;
  } else if (t <= 0.25) {
    r = lerp(68, 59, t * 4);
    g = lerp(1, 82, t * 4);
    b = lerp(84, 139, t * 4);
  } else if (t <= 0.5) {
    r = lerp(59, 33, (t - 0.25) * 4);
    g = lerp(82, 144, (t - 0.25) * 4);
    b = lerp(139, 141, (t - 0.25) * 4);
  } else if (t <= 0.75) {
    r = lerp(33, 93, (t - 0.5) * 4);
    g = lerp(144, 201, (t - 0.5) * 4);
    b = lerp(141, 99, (t - 0.5) * 4);
  } else {
    r = lerp(93, 253, (t - 0.75) * 4);
    g = lerp(201, 231, (t - 0.75) * 4);
    b = lerp(99, 37, (t - 0.75) * 4);
  }
  
  return { r, g, b };
}

function mousePressed() {
  // Restart drawing loop when mouse is pressed to enable interaction
  loop();
}