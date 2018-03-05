/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  std::default_random_engine generator;
  std::normal_distribution<double> x_distribution(x, std[0]);
  std::normal_distribution<double> y_distribution(y, std[1]);
  std::normal_distribution<double> theta_distribution(theta, std[2]);
  particles.clear();
  weights.clear();
  for (int i = 0; i < num_particles; ++i){
    std::vector<int> association;
    std::vector<double> sense_x, sense_y;
    Particle p = {i, x_distribution(generator), y_distribution(generator), theta_distribution(generator),
                         1.0, association, sense_x, sense_y};
    particles.push_back(p);
    weights.push_back(1.0);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  std::vector<Particle> updated_particles;
  std::default_random_engine generator;
  std::normal_distribution<double> x_noise(0, std_pos[0]);
  std::normal_distribution<double> y_noise(0, std_pos[1]);
  std::normal_distribution<double> theta_noise(0, std_pos[2]);
  for(Particle& p : particles){
    if (fabs(yaw_rate) < 0.000001){
      p.x += velocity * delta_t * cos(p.theta);
      p.y += velocity * delta_t * sin(p.theta);
    }
    else {
      p.x += (velocity / yaw_rate) * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
      p.y += (velocity / yaw_rate) * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
    }
    p.x += x_noise(generator);
    p.y += y_noise(generator);
    p.theta += delta_t * yaw_rate + theta_noise(generator);
    p.theta = fmod(p.theta, 2 * M_PI);
  }
  //cout << "Pred done" << endl;
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, const std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

  for (Particle& p : particles) {
    p.sense_x.clear();
    p.sense_y.clear();
    p.associations.clear();
    std::vector<LandmarkObs> transformed;
    for(LandmarkObs observation : observations){
      double transformed_x = p.x + cos(p.theta) * observation.x - sin(p.theta) * observation.y;
      double transformed_y = p.y + sin(p.theta) * observation.x + cos(p.theta) * observation.y;
      LandmarkObs transformed_s = {observation.id, transformed_x, transformed_y};
      transformed.push_back(transformed_s);
      p.sense_x.push_back(transformed_x);
      p.sense_y.push_back(transformed_y);

      int best_landmark_index = 0;
      double best_dist = -1;
      for(int i = 1; i <= predicted.size(); ++i) { // need to start index from i = 1 for autograder to work
        double distance = dist(predicted[i-1].x, predicted[i-1].y, transformed_x, transformed_y);
        if (best_dist < 0 || distance < best_dist) {
          best_dist = distance;
          best_landmark_index = i;
        }
      }
      p.associations.push_back(best_landmark_index);
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
  std::vector<LandmarkObs> predicted;
  for (Map::single_landmark_s landmark_s : map_landmarks.landmark_list) {
    LandmarkObs pred_observation = {landmark_s.id_i, landmark_s.x_f, landmark_s.y_f};
    predicted.push_back(pred_observation);
  }
  dataAssociation(predicted, observations);
  std::vector<double> updated_weights;
  for(Particle& p : particles) {
    double final_weight = 1.0;
    for(int i = 0; i < p.associations.size(); ++i){
      double gauss_norm = (1 / (2 * M_PI * std_landmark[0] * std_landmark[1]));
      double x_diff = std::min(fabs(predicted[p.associations[i]-1].x - p.sense_x[i]), sensor_range);
      double y_diff = std::min(fabs(predicted[p.associations[i]-1].y - p.sense_y[i]), sensor_range);
      double exponent = x_diff * x_diff / (2 * std_landmark[0] * std_landmark[0])
                        + y_diff * y_diff / (2 * std_landmark[1] * std_landmark[1]);
      double weight = gauss_norm * exp(-exponent);
      final_weight *= weight;
    }
    p.weight = final_weight;
    updated_weights.push_back(final_weight);
  }
  weights = updated_weights;
  //cout << "Update done" << endl;
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  std::vector<Particle> resampled_particles;
  std::vector<double> updated_weights;
  std::default_random_engine generator;
  std::discrete_distribution<int> weight_distribution(weights.begin(), weights.end());
  for(int i = 0; i < num_particles; ++i) {
    int index = weight_distribution(generator);
    Particle sample = particles[index];
    resampled_particles.push_back(sample);
    updated_weights.push_back(sample.weight);
  }
  particles = resampled_particles;
  weights = updated_weights;
  //cout << "Resample done" << endl;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
