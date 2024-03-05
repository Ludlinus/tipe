import gzip
import pickle
import random
import time

import neat.checkpoint as checkpointing
import neat.reporting as reporting
from neat.math_util import mean, stdev

import wandb


class WANDB_Reporter(object):
    def __init__(self, intervalle_de_validation=None, fonction_de_validation=None):
        if intervalle_de_validation == None or fonction_de_validation == None:
            self.validation = False
        else:
            self.validation = True
            self.intervalle_de_validation = intervalle_de_validation
            self.fonction_de_validation = fonction_de_validation

    def start_generation(self, generation):
        self.generation = generation

    def end_generation(self, config, population, species_set):
        pass

    def post_evaluate(self, config, population, species, best_genome):
        fitnesses = [c.fitness for c in population.values()]

        log = {
            "epoch": self.generation,
            "fit_mean": mean(fitnesses),
            "fit_std": stdev(fitnesses),
            "nbOfSpecies": len(species.species)
        }

        if self.validation and self.generation % self.intervalle_de_validation == 0:
            fitnesses_validation = self.fonction_de_validation(
                [(genome_id, population[genome_id]) for genome_id in population], config)
            log["validation_fit_mean"] = mean(fitnesses_validation.values())
            log["validation_fit_std"] = stdev(fitnesses_validation.values())
            log["validation_fit_best"] = fitnesses_validation[best_genome.key]

        wandb.log(log)

    def post_reproduction(self, config, population, species):
        pass

    def complete_extinction(self):
        pass

    def found_solution(self, config, generation, best):
        pass

    def species_stagnant(self, sid, species):
        pass

    def info(self, msg):
        pass


class StdOutReporterPeriodique(reporting.BaseReporter):
    """Uses `print` to output information about the run; an example reporter class.
    N'affiche toutes les informations qu'une fois par période"""

    def __init__(self, show_species_detail, periode):
        self.show_species_detail = show_species_detail
        self.generation = None
        self.generation_start_time = None
        self.generation_times = []
        self.num_extinctions = 0
        self.periode = periode
        self.total_stagnant = 0

    def start_generation(self, generation):
        self.generation = generation
        print('\n ****** Running generation {0} ****** \n'.format(generation))
        self.generation_start_time = time.time()

    def end_generation(self, config, population, species_set):
        if self.generation % self.periode == 0:
            ng = len(population)
            ns = len(species_set.species)
            if self.show_species_detail:
                print('Population of {0:d} members in {1:d} species:'.format(ng, ns))
                print("   ID   age  size   fitness   adj fit  stag")
                print("  ====  ===  ====  =========  =======  ====")
                for sid in sorted(species_set.species):
                    s = species_set.species[sid]
                    a = self.generation - s.created
                    n = len(s.members)
                    f = "--" if s.fitness is None else f"{s.fitness:.3f}"
                    af = "--" if s.adjusted_fitness is None else f"{s.adjusted_fitness:.3f}"
                    st = self.generation - s.last_improved
                    print(f"  {sid:>4}  {a:>3}  {n:>4}  {f:>9}  {af:>7}  {st:>4}")
            else:
                print('Population of {0:d} members in {1:d} species'.format(ng, ns))

        elapsed = time.time() - self.generation_start_time
        self.generation_times.append(elapsed)
        self.generation_times = self.generation_times[-10:]
        average = sum(self.generation_times) / len(self.generation_times)
        print('Deleted stagnant : {0:d}'.format(self.total_stagnant))
        self.total_stagnant = 0
        if len(self.generation_times) > 1:
            print("Generation time: {0:.3f} sec ({1:.3f} average)".format(elapsed, average))
        else:
            print("Generation time: {0:.3f} sec".format(elapsed))

    def post_evaluate(self, config, population, species, best_genome):
        # pylint: disable=no-self-use
        if self.generation % self.periode != 0:
            return
        fitnesses = [c.fitness for c in population.values()]
        fit_mean = mean(fitnesses)
        fit_std = stdev(fitnesses)
        best_species_id = species.get_species_id(best_genome.key)
        print('Population\'s average fitness: {0:3.5f} stdev: {1:3.5f}'.format(fit_mean, fit_std))
        print(
            'Best fitness: {0:3.5f} - size: {1!r} - species {2} - id {3}'.format(best_genome.fitness,
                                                                                 best_genome.size(),
                                                                                 best_species_id,
                                                                                 best_genome.key))

    def complete_extinction(self):
        self.num_extinctions += 1
        print('All species extinct.')

    def found_solution(self, config, generation, best):
        print('\nBest individual in generation {0} meets fitness threshold - complexity: {1!r}'.format(
            self.generation, best.size()))

    def species_stagnant(self, sid, species):
        self.total_stagnant += 1

    def info(self, msg):
        pass


class WANDB_Checkpointer(checkpointing.Checkpointer):
    def __init__(self, generation_interval=100, time_interval_seconds=300,
                 filename_prefix='neat-checkpoint-'):
        super().__init__(generation_interval, time_interval_seconds, filename_prefix)

    def save_checkpoint(self, config, population, species_set, generation):
        """ Save the current simulation state. """
        filename = '{0}{1}'.format(self.filename_prefix, generation)
        print("Saving checkpoint to {0}".format(filename))

        with gzip.open(filename, 'w', compresslevel=5) as f:
            data = (generation, config, population, species_set, random.getstate())
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        wandb.save(filename, policy='now')
