#!/usr/bin/python

import os, sys, re
import argparse

import json 

def gen_json(txtfile, jsonfile):
  fields = {} 
  parent_field = ''
  nbr_indent = 0
  nbr_indent_prev = 0
  check_for_dims = False
  with open(txtfile) as fp: 
    for line in fp: 
      mv = re.match(r'HCC clang version\s+(.*)',line)
      if mv:
        key = 'HCCclangversion'
        val = mv.group(1)
        fields[key] = val
        continue
      if check_for_dims == True:
        mc = re.match(r'\s*([x|y|z])\s+(.*)',line) 
        if mc:
          key_sav = mc.group(1)
          if parent_field != '':
            key = parent_field + '_' + mc.group(1)
          else:
            key = mc.group(1)
          val = re.sub(r"\s+", "", mc.group(2))
          #print "key is " + key
          #print "val is " + val
          fields[key] = val
          if key_sav == 'z':
            check_for_dims = False
      nbr_indent_prev = nbr_indent
      mi = re.search(r'^(\s+)\w+', line)
      md = re.search(':', line)
      if mi:
        nbr_indent = len(mi.group(1)) / 2 #indentation cnt
      else:
        if not md:
          tmp = re.sub(r"\s+", "", line)
          if tmp.isalnum():
            #print "changing parent field from " + parent_field + " to " + tmp
            parent_field = tmp
            continue

      if nbr_indent < nbr_indent_prev:
        pos = parent_field.rfind('_')
        if pos != -1:
          #print "changing3 parent field from " + parent_field + " to " + parent_field[:pos]
          parent_field = parent_field[:pos] # remove last _*

      for lin in line.split(';'):
        lin = re.sub(r"\s+", "", lin)
        m = re.match(r'(.*):(.*)', lin)
        if m:
          key, val = m.group(1), m.group(2)
          if parent_field != '':
            key = parent_field + '_' + key
          if val == '':
            mk = re.match(r'.*Dimension',key)
            if mk: # expect x,y,z on next 3 lines
               #print "WILL CHECK FOR DIMS NEXT"
               check_for_dims = True
            #print "changing0 parent field from " + parent_field + " to " + key 
            parent_field = key 
          else:
            #print "key is " + key
            #print "val is " + val
            fields[key] = val
        else:
          #print "Nbr of idents " + str(nbr_indent_prev) + " after " + str(nbr_indent) + "check_for_dims " + str(check_for_dims)
          if nbr_indent != nbr_indent_prev and not check_for_dims :
            #print "changing2 parent field from " + parent_field + " to " + parent_field + '_' + lin.replace(':','')
            parent_field = parent_field + '_' + lin.replace(':','')
  
  outfile = open(jsonfile, "w") 
  json.dump(fields, outfile, indent = 4, sort_keys = False) 
  outfile.close() 

parser = argparse.ArgumentParser(description='parse_text_to_json.py: parses text into json syntax for inclusion into json trace file.')
requiredNamed=parser.add_argument_group('Required arguments')
requiredNamed.add_argument('-in','--in', help='Text to be parsed (such as rocminfo output)', required=True)
requiredNamed.add_argument('-out','--out', help='Output file (.json)', required=True)

args = vars(parser.parse_args())

if __name__ == '__main__':
    gen_json(args['in'],args['out'])

