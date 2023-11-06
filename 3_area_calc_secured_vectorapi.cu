/*
Calculating secure area computation using
available vector API from LITE.
Unfortunately, the final summation that not implemented in LITE
should run in CPU
*/
#include <iostream>
#include <typeinfo>
#include <cmath>

#include <cuda.h>
#include "lite.cu"

#include <chrono>

float* shiftLeft(const float arr[], size_t size) {
    if (size <= 1) {
        float* shifted = new float[size];
        for (size_t i = 0; i < size; i++) shifted[i] = arr[i];
        return shifted;
    }
    float* shifted = new float[size];
    int first_element = arr[0];
    for (size_t i = 0; i < size - 1; i++) shifted[i] = arr[i + 1];
    shifted[size - 1] = first_element; // Wrap around to the first element
    return shifted;
}

int main() {

    
    using std::chrono::high_resolution_clock;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    double avg_time = 0; 
    auto t1 = high_resolution_clock::now();

    int n_points = 299;
    float h_f_x[n_points]; // Host arrays for Float x-coordinates
    float h_f_y[n_points]; // Host arrays for Float y-coordinates 48.555


    // Define the points of a quadrilateral
    h_f_x[0]=10; h_f_y[0]=1;
    h_f_x[1]=10; h_f_y[1]=2;
    h_f_x[2]=10; h_f_y[2]=3;
    h_f_x[3]=10; h_f_y[3]=4;
    h_f_x[4]=10; h_f_y[4]=5;
    h_f_x[5]=10; h_f_y[5]=6;
    h_f_x[6]=10; h_f_y[6]=7;
    h_f_x[7]=10; h_f_y[7]=8;
    h_f_x[8]=10; h_f_y[8]=9;
    h_f_x[9]=10; h_f_y[9]=10;
    h_f_x[10]=10; h_f_y[10]=11;
    h_f_x[11]=10; h_f_y[11]=12;
    h_f_x[12]=10; h_f_y[12]=13;
    h_f_x[13]=10; h_f_y[13]=14;
    h_f_x[14]=10; h_f_y[14]=15;
    h_f_x[15]=10; h_f_y[15]=16;
    h_f_x[16]=10; h_f_y[16]=17;
    h_f_x[17]=10; h_f_y[17]=18;
    h_f_x[18]=10; h_f_y[18]=19;
    h_f_x[19]=10; h_f_y[19]=20;
    h_f_x[20]=10; h_f_y[20]=21;
    h_f_x[21]=10; h_f_y[21]=22;
    h_f_x[22]=10; h_f_y[22]=23;
    h_f_x[23]=10; h_f_y[23]=24;
    h_f_x[24]=10; h_f_y[24]=25;
    h_f_x[25]=10; h_f_y[25]=26;
    h_f_x[26]=10; h_f_y[26]=27;
    h_f_x[27]=10; h_f_y[27]=28;
    h_f_x[28]=10; h_f_y[28]=29;
    h_f_x[29]=10; h_f_y[29]=30;
    h_f_x[30]=10; h_f_y[30]=31;
    h_f_x[31]=10; h_f_y[31]=32;
    h_f_x[32]=10; h_f_y[32]=33;
    h_f_x[33]=10; h_f_y[33]=34;
    h_f_x[34]=10; h_f_y[34]=35;
    h_f_x[35]=10; h_f_y[35]=36;
    h_f_x[36]=10; h_f_y[36]=37;
    h_f_x[37]=10; h_f_y[37]=38;
    h_f_x[38]=10; h_f_y[38]=39;
    h_f_x[39]=10; h_f_y[39]=40;
    h_f_x[40]=10; h_f_y[40]=41;
    h_f_x[41]=10; h_f_y[41]=42;
    h_f_x[42]=10; h_f_y[42]=43;
    h_f_x[43]=10; h_f_y[43]=44;
    h_f_x[44]=10; h_f_y[44]=45;
    h_f_x[45]=10; h_f_y[45]=46;
    h_f_x[46]=10; h_f_y[46]=47;
    h_f_x[47]=10; h_f_y[47]=48;
    h_f_x[48]=10; h_f_y[48]=49;
    h_f_x[49]=10; h_f_y[49]=50;
    h_f_x[50]=10; h_f_y[50]=51;
    h_f_x[51]=10; h_f_y[51]=52;
    h_f_x[52]=10; h_f_y[52]=53;
    h_f_x[53]=10; h_f_y[53]=54;
    h_f_x[54]=10; h_f_y[54]=55;
    h_f_x[55]=10; h_f_y[55]=56;
    h_f_x[56]=10; h_f_y[56]=57;
    h_f_x[57]=10; h_f_y[57]=58;
    h_f_x[58]=10; h_f_y[58]=59;
    h_f_x[59]=10; h_f_y[59]=60;
    h_f_x[60]=10; h_f_y[60]=61;
    h_f_x[61]=10; h_f_y[61]=62;
    h_f_x[62]=10; h_f_y[62]=63;
    h_f_x[63]=10; h_f_y[63]=64;
    h_f_x[64]=10; h_f_y[64]=65;
    h_f_x[65]=10; h_f_y[65]=66;
    h_f_x[66]=10; h_f_y[66]=67;
    h_f_x[67]=10; h_f_y[67]=68;
    h_f_x[68]=10; h_f_y[68]=69;
    h_f_x[69]=10; h_f_y[69]=70;
    h_f_x[70]=10; h_f_y[70]=71;
    h_f_x[71]=10; h_f_y[71]=72;
    h_f_x[72]=10; h_f_y[72]=73;
    h_f_x[73]=10; h_f_y[73]=74;
    h_f_x[74]=10; h_f_y[74]=75;
    h_f_x[75]=10; h_f_y[75]=76;
    h_f_x[76]=10; h_f_y[76]=77;
    h_f_x[77]=10; h_f_y[77]=78;
    h_f_x[78]=10; h_f_y[78]=79;
    h_f_x[79]=10; h_f_y[79]=80;
    h_f_x[80]=10; h_f_y[80]=81;
    h_f_x[81]=10; h_f_y[81]=82;
    h_f_x[82]=10; h_f_y[82]=83;
    h_f_x[83]=10; h_f_y[83]=84;
    h_f_x[84]=10; h_f_y[84]=85;
    h_f_x[85]=10; h_f_y[85]=86;
    h_f_x[86]=10; h_f_y[86]=87;
    h_f_x[87]=10; h_f_y[87]=88;
    h_f_x[88]=10; h_f_y[88]=89;
    h_f_x[89]=10; h_f_y[89]=90;
    h_f_x[90]=10; h_f_y[90]=91;
    h_f_x[91]=10; h_f_y[91]=92;
    h_f_x[92]=10; h_f_y[92]=93;
    h_f_x[93]=10; h_f_y[93]=94;
    h_f_x[94]=10; h_f_y[94]=95;
    h_f_x[95]=10; h_f_y[95]=96;
    h_f_x[96]=10; h_f_y[96]=97;
    h_f_x[97]=10; h_f_y[97]=98;
    h_f_x[98]=10; h_f_y[98]=99;
    h_f_x[99]=10; h_f_y[99]=100;
    h_f_x[100]=11; h_f_y[100]=100;
    h_f_x[101]=12; h_f_y[101]=100;
    h_f_x[102]=13; h_f_y[102]=100;
    h_f_x[103]=14; h_f_y[103]=100;
    h_f_x[104]=15; h_f_y[104]=100;
    h_f_x[105]=16; h_f_y[105]=100;
    h_f_x[106]=17; h_f_y[106]=100;
    h_f_x[107]=18; h_f_y[107]=100;
    h_f_x[108]=19; h_f_y[108]=100;
    h_f_x[109]=20; h_f_y[109]=100;
    h_f_x[110]=21; h_f_y[110]=100;
    h_f_x[111]=22; h_f_y[111]=100;
    h_f_x[112]=23; h_f_y[112]=100;
    h_f_x[113]=24; h_f_y[113]=100;
    h_f_x[114]=25; h_f_y[114]=100;
    h_f_x[115]=26; h_f_y[115]=100;
    h_f_x[116]=27; h_f_y[116]=100;
    h_f_x[117]=28; h_f_y[117]=100;
    h_f_x[118]=29; h_f_y[118]=100;
    h_f_x[119]=30; h_f_y[119]=100;
    h_f_x[120]=31; h_f_y[120]=100;
    h_f_x[121]=32; h_f_y[121]=100;
    h_f_x[122]=33; h_f_y[122]=100;
    h_f_x[123]=34; h_f_y[123]=100;
    h_f_x[124]=35; h_f_y[124]=100;
    h_f_x[125]=36; h_f_y[125]=100;
    h_f_x[126]=37; h_f_y[126]=100;
    h_f_x[127]=38; h_f_y[127]=100;
    h_f_x[128]=39; h_f_y[128]=100;
    h_f_x[129]=40; h_f_y[129]=100;
    h_f_x[130]=41; h_f_y[130]=100;
    h_f_x[131]=42; h_f_y[131]=100;
    h_f_x[132]=43; h_f_y[132]=100;
    h_f_x[133]=44; h_f_y[133]=100;
    h_f_x[134]=45; h_f_y[134]=100;
    h_f_x[135]=46; h_f_y[135]=100;
    h_f_x[136]=47; h_f_y[136]=100;
    h_f_x[137]=48; h_f_y[137]=100;
    h_f_x[138]=49; h_f_y[138]=100;
    h_f_x[139]=50; h_f_y[139]=100;
    h_f_x[140]=51; h_f_y[140]=100;
    h_f_x[141]=52; h_f_y[141]=100;
    h_f_x[142]=53; h_f_y[142]=100;
    h_f_x[143]=54; h_f_y[143]=100;
    h_f_x[144]=55; h_f_y[144]=100;
    h_f_x[145]=56; h_f_y[145]=100;
    h_f_x[146]=57; h_f_y[146]=100;
    h_f_x[147]=58; h_f_y[147]=100;
    h_f_x[148]=59; h_f_y[148]=100;
    h_f_x[149]=60; h_f_y[149]=100;
    h_f_x[150]=61; h_f_y[150]=100;
    h_f_x[151]=62; h_f_y[151]=100;
    h_f_x[152]=63; h_f_y[152]=100;
    h_f_x[153]=64; h_f_y[153]=100;
    h_f_x[154]=65; h_f_y[154]=100;
    h_f_x[155]=66; h_f_y[155]=100;
    h_f_x[156]=67; h_f_y[156]=100;
    h_f_x[157]=68; h_f_y[157]=100;
    h_f_x[158]=69; h_f_y[158]=100;
    h_f_x[159]=70; h_f_y[159]=100;
    h_f_x[160]=71; h_f_y[160]=100;
    h_f_x[161]=72; h_f_y[161]=100;
    h_f_x[162]=73; h_f_y[162]=100;
    h_f_x[163]=74; h_f_y[163]=100;
    h_f_x[164]=75; h_f_y[164]=100;
    h_f_x[165]=76; h_f_y[165]=100;
    h_f_x[166]=77; h_f_y[166]=100;
    h_f_x[167]=78; h_f_y[167]=100;
    h_f_x[168]=79; h_f_y[168]=100;
    h_f_x[169]=80; h_f_y[169]=100;
    h_f_x[170]=81; h_f_y[170]=100;
    h_f_x[171]=82; h_f_y[171]=100;
    h_f_x[172]=83; h_f_y[172]=100;
    h_f_x[173]=84; h_f_y[173]=100;
    h_f_x[174]=85; h_f_y[174]=100;
    h_f_x[175]=86; h_f_y[175]=100;
    h_f_x[176]=87; h_f_y[176]=100;
    h_f_x[177]=88; h_f_y[177]=100;
    h_f_x[178]=89; h_f_y[178]=100;
    h_f_x[179]=90; h_f_y[179]=100;
    h_f_x[180]=91; h_f_y[180]=100;
    h_f_x[181]=92; h_f_y[181]=100;
    h_f_x[182]=93; h_f_y[182]=100;
    h_f_x[183]=94; h_f_y[183]=100;
    h_f_x[184]=95; h_f_y[184]=100;
    h_f_x[185]=96; h_f_y[185]=100;
    h_f_x[186]=97; h_f_y[186]=100;
    h_f_x[187]=98; h_f_y[187]=100;
    h_f_x[188]=99; h_f_y[188]=100;
    h_f_x[189]=100; h_f_y[189]=100;
    h_f_x[190]=101; h_f_y[190]=100;
    h_f_x[191]=102; h_f_y[191]=100;
    h_f_x[192]=103; h_f_y[192]=100;
    h_f_x[193]=104; h_f_y[193]=100;
    h_f_x[194]=105; h_f_y[194]=100;
    h_f_x[195]=106; h_f_y[195]=100;
    h_f_x[196]=107; h_f_y[196]=100;
    h_f_x[197]=108; h_f_y[197]=100;
    h_f_x[198]=109; h_f_y[198]=100;
    h_f_x[199]=110; h_f_y[199]=100;
    h_f_x[200]=110; h_f_y[200]=99;
    h_f_x[201]=110; h_f_y[201]=98;
    h_f_x[202]=110; h_f_y[202]=97;
    h_f_x[203]=110; h_f_y[203]=96;
    h_f_x[204]=110; h_f_y[204]=95;
    h_f_x[205]=110; h_f_y[205]=94;
    h_f_x[206]=110; h_f_y[206]=93;
    h_f_x[207]=110; h_f_y[207]=92;
    h_f_x[208]=110; h_f_y[208]=91;
    h_f_x[209]=110; h_f_y[209]=90;
    h_f_x[210]=110; h_f_y[210]=89;
    h_f_x[211]=110; h_f_y[211]=88;
    h_f_x[212]=110; h_f_y[212]=87;
    h_f_x[213]=110; h_f_y[213]=86;
    h_f_x[214]=110; h_f_y[214]=85;
    h_f_x[215]=110; h_f_y[215]=84;
    h_f_x[216]=110; h_f_y[216]=83;
    h_f_x[217]=110; h_f_y[217]=82;
    h_f_x[218]=110; h_f_y[218]=81;
    h_f_x[219]=110; h_f_y[219]=80;
    h_f_x[220]=110; h_f_y[220]=79;
    h_f_x[221]=110; h_f_y[221]=78;
    h_f_x[222]=110; h_f_y[222]=77;
    h_f_x[223]=110; h_f_y[223]=76;
    h_f_x[224]=110; h_f_y[224]=75;
    h_f_x[225]=110; h_f_y[225]=74;
    h_f_x[226]=110; h_f_y[226]=73;
    h_f_x[227]=110; h_f_y[227]=72;
    h_f_x[228]=110; h_f_y[228]=71;
    h_f_x[229]=110; h_f_y[229]=70;
    h_f_x[230]=110; h_f_y[230]=69;
    h_f_x[231]=110; h_f_y[231]=68;
    h_f_x[232]=110; h_f_y[232]=67;
    h_f_x[233]=110; h_f_y[233]=66;
    h_f_x[234]=110; h_f_y[234]=65;
    h_f_x[235]=110; h_f_y[235]=64;
    h_f_x[236]=110; h_f_y[236]=63;
    h_f_x[237]=110; h_f_y[237]=62;
    h_f_x[238]=110; h_f_y[238]=61;
    h_f_x[239]=110; h_f_y[239]=60;
    h_f_x[240]=110; h_f_y[240]=59;
    h_f_x[241]=110; h_f_y[241]=58;
    h_f_x[242]=110; h_f_y[242]=57;
    h_f_x[243]=110; h_f_y[243]=56;
    h_f_x[244]=110; h_f_y[244]=55;
    h_f_x[245]=110; h_f_y[245]=54;
    h_f_x[246]=110; h_f_y[246]=53;
    h_f_x[247]=110; h_f_y[247]=52;
    h_f_x[248]=110; h_f_y[248]=51;
    h_f_x[249]=110; h_f_y[249]=50;
    h_f_x[250]=110; h_f_y[250]=49;
    h_f_x[251]=110; h_f_y[251]=48;
    h_f_x[252]=110; h_f_y[252]=47;
    h_f_x[253]=110; h_f_y[253]=46;
    h_f_x[254]=110; h_f_y[254]=45;
    h_f_x[255]=110; h_f_y[255]=44;
    h_f_x[256]=110; h_f_y[256]=43;
    h_f_x[257]=110; h_f_y[257]=42;
    h_f_x[258]=110; h_f_y[258]=41;
    h_f_x[259]=110; h_f_y[259]=40;
    h_f_x[260]=110; h_f_y[260]=39;
    h_f_x[261]=110; h_f_y[261]=38;
    h_f_x[262]=110; h_f_y[262]=37;
    h_f_x[263]=110; h_f_y[263]=36;
    h_f_x[264]=110; h_f_y[264]=35;
    h_f_x[265]=110; h_f_y[265]=34;
    h_f_x[266]=110; h_f_y[266]=33;
    h_f_x[267]=110; h_f_y[267]=32;
    h_f_x[268]=110; h_f_y[268]=31;
    h_f_x[269]=110; h_f_y[269]=30;
    h_f_x[270]=110; h_f_y[270]=29;
    h_f_x[271]=110; h_f_y[271]=28;
    h_f_x[272]=110; h_f_y[272]=27;
    h_f_x[273]=110; h_f_y[273]=26;
    h_f_x[274]=110; h_f_y[274]=25;
    h_f_x[275]=110; h_f_y[275]=24;
    h_f_x[276]=110; h_f_y[276]=23;
    h_f_x[277]=110; h_f_y[277]=22;
    h_f_x[278]=110; h_f_y[278]=21;
    h_f_x[279]=110; h_f_y[279]=20;
    h_f_x[280]=110; h_f_y[280]=19;
    h_f_x[281]=110; h_f_y[281]=18;
    h_f_x[282]=110; h_f_y[282]=17;
    h_f_x[283]=110; h_f_y[283]=16;
    h_f_x[284]=110; h_f_y[284]=15;
    h_f_x[285]=110; h_f_y[285]=14;
    h_f_x[286]=110; h_f_y[286]=13;
    h_f_x[287]=110; h_f_y[287]=12;
    h_f_x[288]=110; h_f_y[288]=11;
    h_f_x[289]=110; h_f_y[289]=10;
    h_f_x[290]=110; h_f_y[290]=9;
    h_f_x[291]=110; h_f_y[291]=8;
    h_f_x[292]=110; h_f_y[292]=7;
    h_f_x[293]=110; h_f_y[293]=6;
    h_f_x[294]=110; h_f_y[294]=5;
    h_f_x[295]=110; h_f_y[295]=4;
    h_f_x[296]=110; h_f_y[296]=3;
    h_f_x[297]=110; h_f_y[297]=2;
    h_f_x[298]=110; h_f_y[298]=1;
    
    // 0. Initiate key in CPU
    uchar key[] = { 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00 };
    uint keySize = 16;
    int Nr=10;
    uint e_sched[4*(MAXNR + 1)];
    uint d_sched[4*(MAXNR + 1)];
    makeKey(key, keySize << 3, DIR_BOTH, e_sched, d_sched, Nr);

    float *shifted_h_f_x = shiftLeft(h_f_x, n_points);
    float *shifted_h_f_y = shiftLeft(h_f_y, n_points);

    float *c = new float[n_points];
    float *d = new float[n_points];
    liteMultiplication(c, h_f_x, shifted_h_f_y, n_points, e_sched, d_sched, Nr, 1, n_points);
    liteMultiplication(d, shifted_h_f_x, h_f_y, n_points, e_sched, d_sched, Nr, 1, n_points);
    liteSubstraction(d, c, d, n_points, e_sched, d_sched, Nr, 1, n_points);

    int total = 0;
    for(int i=0;i<n_points;i++){
        total+=d[i];
    }

    auto t2 = high_resolution_clock::now();

    std::cout << "Area: " << total*0.5 << std::endl;

    duration<double, std::milli> ms_double = t2 - t1;
    avg_time += ms_double.count();
    std::cout << avg_time << std::endl;

    return 0;
}
