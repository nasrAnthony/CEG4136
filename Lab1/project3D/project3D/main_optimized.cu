#include <GL/glut.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include <random>    // Pour std::shuffle et std::mt19937 // For std::shuffle and std::mt19937
#include <algorithm> // Pour std::shuffle // For std::shuffle
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define N 1000  // Taille de la grille // Grid size
#define BURN_DURATION 5000  // Durée de combustion d'un arbre en millisecondes (5 secondes) // Tree burning duration in milliseconds (5 seconds)
#define FIRE_START_COUNT 100  // Nombre initial d'incendies // Initial number of fire locations

// Utilisation de vecteurs pour gérer la mémoire // Using vectors to manage memory
std::vector<std::vector<int>> forest(N, std::vector<int>(N, 0));
std::vector<std::vector<int>> burnTime(N, std::vector<int>(N, 0));
bool allBurnedOut = true;  // Indicateur pour vérifier si tous les feux sont éteints // Flag to check if all fires are out

int simulationDuration = 60000;  // Durée de la simulation (60 secondes) // Simulation duration (60 seconds)
int startTime = 0;  // Temps de départ en millisecondes // Start time in milliseconds
int elapsedTime = 0;  // Temps écoulé // Elapsed time
float spreadProbability = 0.3f;  // Probabilité que le feu se propage à un arbre voisin // Probability that fire spreads to a neighboring tree

bool isPaused = false;  // Indicateur de pause // Pause indicator
int pauseStartTime = 0;  // Temps de début de la pause // Start time of pause

float zoomLevel = 1.0f;  // Niveau de zoom // Zoom level
float offsetX = 0.0f, offsetY = 0.0f;  // Décalage horizontal et vertical pour le déplacement // Horizontal and vertical offset for movement
float moveSpeed = 0.05f;  // Vitesse de déplacement de la vue // View movement speed

bool dragging = false;  // Indicateur de glisser-déposer avec la souris // Mouse drag indicator
int lastMouseX, lastMouseY;  // Dernière position de la souris lors du clic // Last mouse position when clicked

// Fonction pour initialiser la forêt // Function to initialize the forest
void initializeForest() {
    // Initialisation de la forêt avec 50% d'arbres // Initializing the forest with 50% trees
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            forest[i][j] = rand() % 2;  // 50% d'arbres (1), 50% vide (0) // 50% trees (1), 50% empty space (0)
            burnTime[i][j] = 0;  // Aucun arbre ne brûle au départ // No tree is burning at the start
        }
    }

    // Liste de positions disponibles pour allumer les feux // List of available positions to start fires
    std::vector<std::pair<int, int>> availablePositions;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (forest[i][j] == 1) {  // Ajouter les positions avec des arbres dans la liste // Add positions with trees to the list
                availablePositions.push_back({ i, j });
            }
        }
    }

    // Mélanger les positions disponibles pour une distribution plus uniforme // Shuffle the available positions for a more uniform distribution
    std::random_device rd;  // Générateur de nombres aléatoires basé sur l'implémentation du système // Random number generator based on system implementation
    std::mt19937 g(rd());   // Générateur de nombres pseudo-aléatoires basé sur Mersenne Twister // Mersenne Twister-based pseudo-random number generator
    std::shuffle(availablePositions.begin(), availablePositions.end(), g);

    // Allumer des feux de manière uniforme sur la grille // Ignite fires uniformly across the grid
    for (int fire = 0; fire < FIRE_START_COUNT && !availablePositions.empty(); fire++) {
        int fireX = availablePositions[fire].first;
        int fireY = availablePositions[fire].second;

        forest[fireX][fireY] = 2;  // Allumer l'arbre en feu // Ignite the tree
        burnTime[fireX][fireY] = BURN_DURATION;  // Définir le temps de combustion // Set the burn duration
    }

    startTime = glutGet(GLUT_ELAPSED_TIME);  // Réinitialiser le temps de départ // Reset start time
    elapsedTime = 0;  // Réinitialiser le temps écoulé // Reset elapsed time
    isPaused = false;  // Fin de la pause // End of pause
}

// Fonction d'initialisation OpenGL // OpenGL initialization function
void initGL() {
    glClearColor(1.0, 1.0, 1.0, 1.0);  // Couleur de fond blanche // White background color
    glEnable(GL_DEPTH_TEST);  // Activer le test de profondeur // Enable depth test
}

// Fonction pour dessiner la grille // Function to draw the grid
void drawForest() {
    float cellSize = 2.0f / N;  // Taille de chaque cellule ajustée par la taille N // Adjusted cell size based on grid size N

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            // Choisir la couleur en fonction de l'état de la cellule // Set color based on the state of the cell
            if (forest[i][j] == 0 && burnTime[i][j] == 0) {
                glColor3f(0.8f, 0.8f, 0.8f);  // Espace vide (gris) // Empty space (gray)
            }
            else if (forest[i][j] == 1) {
                glColor3f(0.0f, 1.0f, 0.0f);  // Arbre (vert) // Tree (green)
            }
            else if (forest[i][j] == 2) {
                glColor3f(1.0f, 0.0f, 0.0f);  // Arbre en feu (rouge) // Tree on fire (red)
            }
            else if (forest[i][j] == 3) {
                glColor3f(0.0f, 0.0f, 0.0f);  // Arbre brûlé (noir) // Burned tree (black)
            }

            // Dessiner la cellule // Draw the cell
            float x = -1.0f + j * cellSize;
            float y = -1.0f + i * cellSize;
            glBegin(GL_QUADS);
            glVertex2f(x, y);
            glVertex2f(x + cellSize, y);
            glVertex2f(x + cellSize, y + cellSize);
            glVertex2f(x, y + cellSize);
            glEnd();
        }
    }
}

//defining main kernel
__global__ __host__ void updateForestKernel(int* forest, int* burnTime, int gridDims, float spreadProbability, bool allBurnedFlag) {
    //implementation of kernel
    //calculate the global thread index by converting from 2D -> 1D array
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / gridDims;
    int col = idx % gridDims;
    if (forest[idx] == 2) {  // Si l'arbre est en feu // If the tree is on fire
        burnTime[idx] -= 200;  // Réduire le temps de combustion // Reduce the burning time

        // Vérifier si le feu est éteint // Check if the fire is out
        if (burnTime[idx] <= 0) {
            forest[idx] = 3;  // Marquer l'arbre comme brûlé // Mark the tree as burned
        }
        else {
            // Propagation du feu aux voisins // Propagation of fire to neighbors
            if (row > 0 && forest[idx - gridDims] == 1 && (rand() / (float)RAND_MAX) < spreadProbability) {
                forest[idx - gridDims] = 2;
                burnTime[idx - gridDims] = BURN_DURATION;
            }
            if (row < gridDims - 1 && forest[idx] == 1 && (rand() / (float)RAND_MAX) < spreadProbability) {
                forest[idx + gridDims] = 2;
                burnTime[idx + gridDims] = BURN_DURATION;
            }
            if (col > 0 && forest[idx - 1] == 1 && (rand() / (float)RAND_MAX) < spreadProbability) {
                forest[idx - 1] = 2;
                burnTime[idx - 1] = BURN_DURATION;
            }
            if (col < gridDims - 1 && forest[idx] == 1 && (rand() / (float)RAND_MAX) < spreadProbability) {
                forest[idx + 1] = 2;
                burnTime[idx + 1] = BURN_DURATION;
            }
        }
        if (forest [idx] == 2) {
            allBurnedOut = false;
        }
    }
}

// Fonction pour mettre à jour la forêt et la propagation du feu // Function to update the forest and fire propagation
void updateForest() {
    if (isPaused) {  // Si la simulation est en pause, réinitialiser la forêt après la pause // If the simulation is paused, reset the forest after the pause
        if (glutGet(GLUT_ELAPSED_TIME) - pauseStartTime >= 3000) {
            initializeForest();  // Réinitialiser la forêt après 3 secondes // Reset the forest after 3 seconds
        }
        return;
    }

    
    std::vector<std::vector<int>> newForest = forest;  // Copie la forêt actuelle // Copy the current forest

    //create pointer initial pointer to vector.
    int* dev_forest;
    int* dev_burn_time;

    std::vector<int> flatForest;
    std::vector<int> flatBurnTime;
    for (int i = 0; i < N; i++) {
        flatForest.insert(flatForest.end(), forest[i].begin(), forest[i].end());
        flatBurnTime.insert(flatBurnTime.end(), burnTime[i].begin(), burnTime[i].end());
    }


    //allocate memory space on device to hold forest.
    cudaMalloc((void**)&dev_forest, N * N * sizeof(int));
    //allocate memory space on device to hold burn time grid.
    cudaMalloc((void**)&dev_burn_time, N * N * sizeof(int));

    //copy the new flat grids to the device memory
    cudaMemcpy(dev_forest, flatForest.data(), N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_burn_time, flatBurnTime.data(), N * N * sizeof(int), cudaMemcpyHostToDevice);



    //define parameters for further optimization/testing. 
    const int numBlocks = 256; //good for warping? 
    const int numThreads = (N * N) / numBlocks; // 3907?/block

    //call kernel from host code
   
    updateForestKernel <<< numBlocks, numThreads >>> (dev_forest, dev_burn_time, N, spreadProbability);
        
    cudaMemcpy(flatForest.data(), dev_forest, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(flatBurnTime.data(), dev_forest, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        std::copy(flatForest.begin() + i * N, flatForest.begin() + (i + 1) * N, forest[i].begin());
        std::copy(flatBurnTime.begin() + i * N, flatBurnTime.begin() + (i + 1) * N, burnTime[i].begin());
    }

    if (allBurnedOut) {  // Si tous les feux sont éteints, mettre la simulation en pause // If all fires are out, pause the simulation
        isPaused = true;
        pauseStartTime = glutGet(GLUT_ELAPSED_TIME);
    }
}

// Fonction d'affichage // Display function
void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  // Effacer le tampon de couleur et de profondeur // Clear color and depth buffer
    glLoadIdentity();  // Réinitialiser la matrice modèle-vue // Reset the model-view matrix
    glTranslatef(offsetX, offsetY, 0.0f);  // Appliquer le décalage // Apply translation offset
    glScalef(zoomLevel, zoomLevel, 1.0f);  // Appliquer le zoom // Apply zoom
    drawForest();  // Dessiner la forêt // Draw the forest
    glutSwapBuffers();  // Échanger les tampons pour afficher l'image // Swap buffers to display the image
}

// Fonction pour animer la simulation // Function to animate the simulation
void update(int value) {
    updateForest();  // Mettre à jour la forêt à chaque cycle // Update the forest at each cycle
    glutPostRedisplay();  // Demander un nouveau rendu // Request a new rendering
    glutTimerFunc(200, update, 0);  // Programmer la prochaine mise à jour dans 200 ms // Schedule the next update in 200 ms
}

// Gestion du clavier pour zoomer/dézoomer et réinitialiser // Keyboard handling for zooming and resetting
void keyboard(unsigned char key, int x, int y) {
    switch (key) {
    case '+':
        zoomLevel *= 1.1f;  // Augmenter le niveau de zoom // Increase zoom level
        break;
    case '-':
        zoomLevel /= 1.1f;  // Diminuer le niveau de zoom // Decrease zoom level
        if (zoomLevel < 0.1f) zoomLevel = 0.1f;
        break;
    case 'r':  // Touche pour réinitialiser // Reset key
        zoomLevel = 1.0f;  // Réinitialiser le zoom et le décalage // Reset zoom and offset
        offsetX = 0.0f;
        offsetY = 0.0f;
        break;
    case 27:  // Touche Échap pour quitter // Escape key to quit
        exit(0);
    }
    glutPostRedisplay();  // Redessiner la scène // Redraw the scene
}

// Gestion des touches fléchées pour déplacer la vue // Arrow keys handling for moving the view
void specialKeys(int key, int x, int y) {
    switch (key) {
    case GLUT_KEY_UP:
        offsetY += moveSpeed / zoomLevel;  // Déplacer la vue vers le haut // Move the view up
        break;
    case GLUT_KEY_DOWN:
        offsetY -= moveSpeed / zoomLevel;  // Déplacer la vue vers le bas // Move the view down
        break;
    case GLUT_KEY_LEFT:
        offsetX += moveSpeed / zoomLevel;  // Déplacer la vue vers la gauche // Move the view left
        break;
    case GLUT_KEY_RIGHT:
        offsetX -= moveSpeed / zoomLevel;  // Déplacer la vue vers la droite // Move the view right
        break;
    }
    glutPostRedisplay();  // Redessiner la scène // Redraw the scene
}

// Gestion de la souris pour déplacer la vue // Mouse handling for moving the view
void mouseMotion(int x, int y) {
    if (dragging) {
        offsetX += (x - lastMouseX) * moveSpeed / zoomLevel;  // Mettre à jour le décalage horizontal // Update horizontal offset
        offsetY -= (y - lastMouseY) * moveSpeed / zoomLevel;  // Mettre à jour le décalage vertical // Update vertical offset
        lastMouseX = x;
        lastMouseY = y;
        glutPostRedisplay();  // Redessiner la scène // Redraw the scene
    }
}

// Fonction pour gérer le clic de souris // Function to handle mouse clicks
void mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON) {  // Si le bouton gauche de la souris est enfoncé // If the left mouse button is pressed
        if (state == GLUT_DOWN) {
            dragging = true;
            lastMouseX = x;
            lastMouseY = y;
        }
        else {
            dragging = false;
        }
    }
}

// Fonction principale // Main function
int main(int argc, char** argv) {
    srand(static_cast<unsigned>(time(NULL)));  // Initialiser le générateur de nombres aléatoires // Initialize random number generator
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(800, 800);
    glutCreateWindow("Simulation de feux de forêt/Forest Fire Simulation");  // Créer la fenêtre OpenGL // Create the OpenGL window

    initGL();
    initializeForest();

    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutSpecialFunc(specialKeys);
    glutMouseFunc(mouse);
    glutMotionFunc(mouseMotion);
    glutTimerFunc(200, update, 0);

    glutMainLoop();
    return 0;
}
