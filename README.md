# ML in a Box

Last updated: Sep 16th 2021

This is to update the Paperspace Core "ML-in-a-Box" template VM image from Ubuntu 18.04 to 20.04.

By recording the software choice and scripts used to set up the new template, this should make it easier to update in future, and keep the VM up to date with an appropriate base of ML software for users. (The previous 18.04 template has no record of what was run to create it.)

This also makes us flexible and open to customer feedback, as the script can be straightforwardly altered to add new tools, or remove existing ones, and rerun.

## Who is this for?

We assume a generic advanced data science user who probably wants GPU access, but not any particular specialized subfield of data science such as computer vision or natural language processing. Such users can build upon this base to create their own stack, or we can create other VMs for subfields, similar to what can be done with Gradient containers.

We assume they have access to machines outside of this VM, so non-data-science software used everyday by many people is not included.

Some particular software choices are also influenced by assumed details about a user, and are mentioned in the two tables below.

## Software included

Currently we plan to install the following data science software:

**TODO**: Table

## Software not included

Other software considered but not included.

The potential data science stack is far larger than any one person will use, for example, the Anaconda Python distribution for data science has over 7500 optional packages, so we don't attempt to cover everything here.

Some generic categories of software not included:

 - Non-data-science software
 - Commercial software
 - Software only used in particular specialized data science subfields (although we assume our users probably want a GPU)

**TODO**: Table

## Script

**TODO**: Add script and how to run it on the VM to install the software and make the VM a template
