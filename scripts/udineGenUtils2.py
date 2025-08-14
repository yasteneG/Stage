#!/usr/local/bin/env python3
# -*- coding: utf-8 -*-
# $Id: udineGenUtils2.py 14 2010-02-05 04:05:12Z leo.monash.uni $
# ...
"""
You can get the code, as well as information on which papers to cite
if you use this code, at http://code.google.com/p/udinettgen

To run, make sure that combinedITC07graphs.pickle is in the same
directory, and run 'python udinettgen2.py 100 70' where 
100 is the number of events (lectures across courses) and
70 is the occupancy (rooms*periods - lectures).
There is also an example of how to use the classes in your own code
at the bottom of this file, along with a comprehensive unit test if
you have a bash shell available.
"""

import math
import sys
from math import ceil
from datetime import datetime
from random import seed, random, choice, randrange, getstate, setstate, randint, uniform, sample
from optparse import OptionParser

parser = OptionParser()

real_world_avg = {
    "n_rooms" : "?",
    "n_courses" : "?",
    "lectures" : 414,
    "days": 5.11,
    "periods_per_day": 6.72,
    "min_lects_per_course": 1.85,
    "max_lects_per_course": 6.48,
    "teachers": 102.28,
    "min_courses_per_teacher": 1.00,
    "max_courses_per_teacher": 4.61,
    "curricula": 363.82,
    "min_courses_per_curriculum": 1.62,
    "max_courses_per_curriculum": 8.56,
    "min_room_size": 23.30,
    "max_room_size": 367.16,
    "constraints": 1491.59,
}

def help_str(dest, help_text):
    return help_text + ' (real world avg: ' + str(real_world_avg[dest]) + ')'

parser.add_option("--out", dest="out", type="string", 
    help="output file name")
parser.add_option("--name", dest="name", type="string", 
    help="instances id")
parser.add_option("--n_courses", dest="n_courses", type="int", 
    help=help_str("n_courses", "")) 
parser.add_option("--lectures", dest="lectures", type="int", 
    help=help_str("lectures", "")) 
parser.add_option("--n_rooms", dest="n_rooms", type="int", 
    help=help_str("n_rooms", "")) 
parser.add_option("--days", dest="days", type="int", 
    help=help_str("days", "days per week"))
parser.add_option("--periods", dest="periods_per_day", type="int", 
    help=help_str("periods_per_day", "periods per day"))
parser.add_option("--min_lects_per_course", dest="min_lects_per_course", type="int", 
    help=help_str("min_lects_per_course","min lectures per course (soft)"))
parser.add_option("--max_lects_per_course", dest="max_lects_per_course", type="int", 
    help=help_str("max_lects_per_course", "max lectures per course (hard)"))
parser.add_option("--teachers", dest="teachers", type="int",  
    help=help_str("teachers", "number of teachers"))
parser.add_option("--min_courses_per_teacher", dest="min_courses_per_teacher", type="int",  
    help=help_str("min_courses_per_teacher", "min courses per teachers"))
parser.add_option("--max_courses_per_teacher", dest="max_courses_per_teacher", type="int",  
    help=help_str("max_courses_per_teacher", "max courses per teachers"))
parser.add_option("--curricula", dest="curricula", type="int",  
    help=help_str("curricula", "number of curricula"))
parser.add_option("--min_courses_per_curriculum", dest="min_courses_per_curriculum", type="int",  
    help=help_str("min_courses_per_curriculum", "min courses per curriculum"))
parser.add_option("--max_courses_per_curriculum", dest="max_courses_per_curriculum", type="int", 
    help=help_str("max_courses_per_curriculum", "max courses per curriculum"))
parser.add_option("--min_room_size", dest="min_room_size", type="int", 
    help=help_str("min_room_size", "min room size"))
parser.add_option("--max_room_size", dest="max_room_size", type="int", 
    help=help_str("max_room_size", "max room size"))
parser.add_option("--constraints", dest="constraints", type="int",
    help=help_str("constraints", "number of unavailability constraints"))
parser.add_option("--gen-ctt", dest="gen_ctt", action="store_true", default=False,
    help="generate the old ctt format instead of the new ectt fmt.")
parser.add_option("--seed", dest="seed", type="int",
    help="random seed")


def usage(): 
    parser.print_help()
    sys.exit(-1)

class TimetablingInstance:
    """A class storing properties of a (randomly generated) instance of a course timetabling problem."""

    @staticmethod
    def from_argv():
        (opts, args) = parser.parse_args()
        for x in dir(opts):
            if not x.startswith("_") and not x == "seed" and getattr(opts, x) == None:
                usage()
        return TimetablingInstance(opts)

    def __init__(self, opts):
        if opts.seed:
            seed(opts.seed)
        self.opts = opts
        self.intRoomSizes = []
        self.intCourseIds = []
        # For safety, ensure min_lects_per_course >= 1
        if self.opts.min_lects_per_course < 1:
            print("min_lects_per_course must be greater than 0")
            self.opts.min_lects_per_course = 1

    def genRooms(self):
        outRooms = []
        outRooms.append("\nROOMS:\n")
        distinct = []
        # Generate a set of room capacities
        while len(distinct) < self.opts.n_rooms:
            newSize = randrange(self.opts.min_room_size, self.opts.max_room_size)
            distinct.append(newSize)
        distinct.sort()

        self.intRoomSizes = distinct[:]
        for room_idx, capacity in enumerate(distinct, start=1):
            if self.opts.gen_ctt:
                outRooms.append(f"R{room_idx:02d} {capacity}\n")
            else:
                # The '0' at the end might be for some additional field in ectt format
                outRooms.append(f"R{room_idx:02d} {capacity} 0\n")

        return outRooms

    def genCourses(self):
        outCourses = []
        outCourses.append("\nCOURSES:\n")

        left_lects = self.opts.lectures - self.opts.n_courses
        self.courses = []

        # 1) Génération du nombre de lectures par cours
        for i in range(self.opts.n_courses):
            # On prend randrange((min_lects_per_course - 1), max_lects_per_course)
            # puis on borne par left_lects
            lects = 1 + min(
                randrange(self.opts.min_lects_per_course - 1, self.opts.max_lects_per_course),
                max(left_lects, 0)
            )
            max_mindays = min(lects, self.opts.days) + 1
            self.courses.append({
                "id": f"Course{i:03d}",
                "lects": lects,
                "min_days": randrange(1, max_mindays)
            })
            left_lects -= (lects - 1)

        # 2) Distribution des profs (teachers)
        #    En Python 3, range(...) n'est pas une liste, d'où list(range(...)) * ...
        taught = list(range(self.opts.teachers)) * self.opts.min_courses_per_teacher
        teachers = set(range(self.opts.teachers))

        # Tant qu'on n'a pas assez de teachers dans 'taught' pour couvrir tous les cours
        while len(taught) < self.opts.n_courses:
            teacher = teachers.pop()
            # Ajout de prof dans taught
            # On prend un max(...) pour savoir combien de fois on l'ajoute
            # (code d'origine un peu cryptique, on le garde tel quel)
            for _ in range(max(
                self.opts.max_courses_per_teacher - self.opts.min_courses_per_teacher,
                len(taught) - self.opts.n_courses
            )):
                taught.append(teacher)

        # 3) Génération du nombre d'étudiants
        #    On veut un minStu, un maxStu, etc.
        minStu = int(self.opts.min_room_size / 2)
        maxStu = self.intRoomSizes[-1]  # la + grande room
        if len(self.intRoomSizes) > 1:
            maxStu = self.intRoomSizes[-2]  # la 2ème + grande room

        students = []
        for _ in range(self.opts.n_courses):
            students.append(randrange(minStu, maxStu))

        # 4) On écrit la ligne de chaque cours
        for course, teacher, capacity in zip(self.courses, taught, students):
            self.intCourseIds.append(course["id"])
            if self.opts.gen_ctt:
                outCourses.append(
                    f"{course['id']} t{teacher:02d} {course['lects']} {course['min_days']} {capacity}\n"
                )
            else:
                outCourses.append(
                    f"{course['id']} t{teacher:02d} {course['lects']} {course['min_days']} {capacity} 0\n"
                )

        return outCourses

    def genConst(self):
        outUnavail = []
        outUnavail.append("\nUNAVAILABILITY_CONSTRAINTS:\n")
        unavail = set()

        while len(unavail) < self.opts.constraints:
            course = choice(self.intCourseIds)
            day = randrange(0, self.opts.days)
            period = randrange(0, self.opts.periods_per_day)
            unavail.add((course, day, period))

        for course, day, period in unavail:
            outUnavail.append(f"{course} {day} {period}\n")

        if not self.opts.gen_ctt:
            outUnavail.append("\nROOM_CONSTRAINTS:\n")

        return outUnavail

    def genCurricula(self):
        outCurr = []
        outCurr.append("\nCURRICULA:\n")
        curr = []

        # 1) On crée un certain nombre de curricula,
        #    chacun contenant un échantillon de cours.
        for i in range(self.opts.curricula):
            nCourses = randrange(self.opts.min_courses_per_curriculum, self.opts.max_courses_per_curriculum + 1)
            courses_in_curr = sample(self.courses, nCourses)
            curr.append((i, courses_in_curr))

        self.opts.curricula = len(curr)

        # 2) On parcourt la liste des curriculums générés.
        for currid, courseset in curr:
            # Ici, on doit trier par un critère clair, ex. l'ID du cours
            courseset = sorted(courseset, key=lambda c: c["id"])
            line = (
                f"Curr{currid+1:03d} {len(courseset)} "
                + " ".join(course["id"] for course in courseset)
                + "\n"
            )
            outCurr.append(line)

        return outCurr

    def genHeader(self):
        outHeader = []
        outHeader.append(f"Name: {self.opts.name}\n")
        outHeader.append(f"Courses: {self.opts.n_courses}\n")
        outHeader.append(f"Rooms: {self.opts.n_rooms}\n")
        outHeader.append(f"Days: {self.opts.days}\n")
        outHeader.append(f"Periods_per_day: {self.opts.periods_per_day}\n")
        outHeader.append(f"Curricula: {self.opts.curricula}\n")

        if self.opts.gen_ctt:  
            outHeader.append(f"Constraints: {self.opts.constraints}\n")
        else:  
            outHeader.append(f"Min_Max_Daily_Lectures: 0 {self.opts.periods_per_day}\n")
            outHeader.append(f"UnavailabilityConstraints: {self.opts.constraints}\n")
            outHeader.append("RoomConstraints: 0\n")

        return outHeader

    def genInstanceDef(self):
        # 1) Rooms
        rooms = self.genRooms()
        # 2) Courses
        courses = self.genCourses()
        # 3) Curricula
        curr = self.genCurricula()
        # 4) Constraints
        const = self.genConst()
        # 5) Header
        header = self.genHeader()

        return header + courses + rooms + curr + const + ["\nEND."]

def choiceWithWeights(elemFreqTuples):
    """Like choice but takes into account weights."""
    total_weight = sum(f for (_, f) in elemFreqTuples)
    X = random() * total_weight
    s = 0.0
    for (i, w) in elemFreqTuples:
        s += w
        if X < s:
            return i
    # fallback (should not happen)
    return elemFreqTuples[-1][0]


if __name__ == '__main__':
    instance = TimetablingInstance.from_argv()
    inst_lns = instance.genInstanceDef()
    (opts, args) = parser.parse_args()
    with open(opts.out, "w") as f:
        for line in inst_lns:
            f.write(line)
