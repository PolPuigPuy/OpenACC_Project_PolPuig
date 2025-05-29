/*
 *  training.c
 *
 *  Arxiu reutilitzat de l'assignatura de Computació d'Altes Prestacions de
 * l'Escola d'Enginyeria de la Universitat Autònoma de Barcelona Created on: 31
 * gen. 2019 Last modified: Check git. Author: cguzman
 *  Modified: Blanca Llauradó, Christian Germer
 *
 *  Descripció:
 *  Funcions per entrenar la xarxa neuronal.
 */

#include "training.h"

#include <math.h>

/**
 * @brief Inicialitza la capa incial de la xarxa (input layer) amb l'entrada
 *que volem reconeixer.
 *
 * @param i Índex de l'element del conjunt d'entrenament que farem servir.
 **/
void feed_input(int i) {
  // This function is parallelized to give an example of how to manage
  // parallelize the rest of the training functions.
#pragma acc parallel loop present(lay, input, num_neurons[0]) copyin(i)
  for (int j = 0; j < num_neurons[0]; j++)
    lay[0].actv[j] = input[i][j];
  // HINT: It is recommended to continue with the next easier function
  // (update_weights).
}

/**
 * @brief Propagació dels valors de les neurones de l'entrada (valors a la input
 * layer) a la resta de capes de la xarxa fins a obtenir una predicció (sortida)
 *
 * @details La capa d'entrada (input layer = capa 0) ja ha estat inicialitzada
 * amb els valors de l'entrada que volem reconeixer. Així, el for més extern
 * (sobre i) recorre totes les capes de la xarxa a partir de la primera capa
 * hidden (capa 1). El for intern (sobre j) recorre les neurones de la capa i
 * calculant el seu valor d'activació [lay[i].actv[j]]. El valor d'activació de
 * cada neurona depén de l'exitació de la neurona calculada en el for més intern
 * (sobre k) [lay[i].z[j]]. El valor d'exitació s'inicialitza amb el biax de la
 * neurona corresponent [j] (lay[i].bias[j]) i es calcula multiplicant el valor
 *          d'activació de les neurones de la capa anterior (i-1) pels pesos de
 * les connexions (out_weights) entre les dues capes. Finalment, el valor
 * d'activació de la neurona (j) es calcula fent servir la funció RELU
 * (REctified Linear Unit) si la capa (j) és una capa oculta (hidden) o la
 * funció Sigmoid si es tracte de la capa de sortida.
 *
 */
void forward_prop() {
  for (int i = 1; i < num_layers; ++i) {
    #pragma acc parallel loop present(lay, num_neurons)
    for (int j = 0; j < num_neurons[i]; ++j) {
      float temp = lay[i].bias[j];

      #pragma acc loop vector reduction(+:temp)
      for (int k = 0; k < num_neurons[i-1]; ++k) {
        temp += lay[i-1].out_weights[j * num_neurons[i-1] + k]
              * lay[i-1].actv[k];
      }

      lay[i].z[j] = temp;
      if (i < num_layers - 1)
        lay[i].actv[j] = (temp < 0.0f ? 0.0f : temp);
      else
        lay[i].actv[j] = 1.0f / (1.0f + expf(-temp));
    }
  }
}


/**
 * @brief Calcula el gradient que es necessari aplicar als pesos de les
 * connexions entre neurones per corregir els errors de predicció
 *
 * @details Calcula dos vectors de correcció per cada capa de la xarxa, un per
 * corregir els pesos de les connexions de la neurona (j) amb la capa anterior
 *          (lay[i-1].dw[j]) i un segon per corregir el biax de cada neurona de
 * la capa actual (lay[i].bias[j]). Hi ha un tractament diferent per la capa de
 * sortida (num_layesr -1) perquè aquest és l'única cas en el que l'error es
 * conegut (lay[num_layers-1].actv[j] - desired_outputs[p][j]). Això es pot
 * veure en els dos primers fors. Per totes les capes ocultes (hidden layers) no
 * es pot saber el valor d'activació esperat per a cada neurona i per tant es fa
 * una estimació. Aquest càlcul es fa en el doble for que recorre totes les
 * capes ocultes (sobre i) neurona a neurona (sobre j). Es pot veure com en cada
 * cas es fa una estimació de quines hauríen de ser les activacions de les
 * neurones de la capa anterior (lay[i-1].dactv[k] = lay[i-1].out_weights[j*
 * num_neurons[i-1] + k] * lay[i].dz[j];), excepte pel cas de la capa d'entrada
 * (input layer) que és coneguda (imatge d'entrada).
 *
 */
void back_prop(int p) {
  // Output Layer

  for (int j = 0; j < num_neurons[num_layers - 1]; j++) {
    lay[num_layers - 1].dz[j] =
        (lay[num_layers - 1].actv[j] - desired_outputs[p][j]) *
        (lay[num_layers - 1].actv[j]) * (1 - lay[num_layers - 1].actv[j]);
    lay[num_layers - 1].dbias[j] = lay[num_layers - 1].dz[j];
  }

  // This code has been re-organized to allow for using collapse(2). Could you
  // tell if that is the fastest approach?
  for (int j = 0; j < num_neurons[num_layers - 1]; j++) {
    for (int k = 0; k < num_neurons[num_layers - 2]; k++) {
      lay[num_layers - 2].dw[j * num_neurons[num_layers - 2] + k] =
          lay[num_layers - 1].dz[j] * lay[num_layers - 2].actv[k];
    }
  }
  for (int k = 0; k < num_neurons[num_layers - 2]; k++) {
    lay[num_layers - 2].dactv[k] =
        lay[num_layers - 2].out_weights[(num_neurons[num_layers - 1] - 1) *
                                            num_neurons[num_layers - 2] +
                                        k] *
        lay[num_layers - 1].dz[num_neurons[num_layers - 1] - 1];
  }

  // Hidden Layers
  // This code has been re-organized to allow for using collapse(2). Could you
  // tell if that is the fastest approach?
  for (int i = num_layers - 2; i > 0; i--) {
    for (int j = 0; j < num_neurons[i]; j++) {
      lay[i].dz[j] = (lay[i].z[j] >= 0) ? lay[i].dactv[j] : 0;
      lay[i].dbias[j] = lay[i].dz[j];
    }
    for (int j = 0; j < num_neurons[i]; j++) {
      for (int k = 0; k < num_neurons[i - 1]; k++) {
        lay[i - 1].dw[j * num_neurons[i - 1] + k] =
            lay[i].dz[j] * lay[i - 1].actv[k];
        if (i > 1) // Never runs because of using 3 layers
          lay[i - 1].dactv[k] =
              lay[i - 1].out_weights[j * num_neurons[i - 1] + k] * lay[i].dz[j];
      }
    }
  }
}

/**
 * @brief Actualitza els vectors de pesos (out_weights) i de biax (bias) de
 * cada etapa d'acord amb els càlculs fet a la funció de back_prop i el factor
 * d'aprenentatge alpha
 *
 * @see back_prop
 */
void update_weights(void) {
  // HINT: Parallelize this after feed_input.
  // Notice there are variables used here that has some new values
  // from "forward_prop" and "backward_prop". Since these functions are in the
  // CPU, and this function is in the GPU, these variables needs to be updated
  // in the GPU. You can do that with "update device". For example:
  // for (int i = 0; i < num_layers - 1; i++){
  //  #pragma acc update device(lay[i].dw[0 : num_neurons[i])
  // }
  
  // All of these arrays are now present() on the device:
  //   lay, num_neurons, alpha
  #pragma acc parallel present(lay, num_neurons, alpha)
  {
    for (int i = 0; i < num_layers - 1; i++) {
      // update all the out_weights for layer i in parallel
      #pragma acc loop collapse(2)
      for (int j = 0; j < num_neurons[i+1]; j++)
        for (int k = 0; k < num_neurons[i]; k++)
          lay[i].out_weights[j * num_neurons[i] + k] -=
            alpha * lay[i].dw[j * num_neurons[i] + k];

      // update all the biases for layer i in parallel
      #pragma acc loop
      for (int j = 0; j < num_neurons[i+1]; j++)
        lay[i].bias[j] -= alpha * lay[i].dbias[j];
    }
  }
  for (int i = 0; i < num_layers - 1; i++) {
    #pragma acc update host(lay[i].out_weights[0:num_neurons[i+1] * num_neurons[i]])
    #pragma acc update host(lay[i].bias[0:num_neurons[i+1]])
  }
}
